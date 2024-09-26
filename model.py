# ====================================================
# Script Name: model.py
# Description: Defines the main objects required to use and infer parameters in discretized ARNICA plant
# growth model 
# Author: Tom Guédon
# Date: 2024-09
# Version: 1.0
#
# Python Version: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
# Système d'exploitation: Windows 10
#
# Contact : tom.guedondurieu@gmail.com
# ====================================================

# pylint: disable= C0116, C0103, R0913

from dataclasses import dataclass
from typing import NamedTuple
import parametrization_cookbook.jax as pc
import jax.numpy as jnp


class Parameter(NamedTuple):
    """
    Create an object Parameter, that has a default value, and a parametrization.
    this object is only required to define the parameter spaces (parametrizations)
    the default value is the one used for the non estimated parameters
    """

    default_value: float
    parametrization: pc.Param


class _ParamsModelBase(NamedTuple):
    """
    _ParamsModelBase object defines a parameter object, whose attributes are float of Parameter(NamedTuple)
    It is not supposed to be used, as one only require the two versions (only float or only Parameter for definition)
    """

    SNU: Parameter | float
    R_RGRmax: Parameter | float
    R_NCont_min: Parameter | float
    remob_NStor: Parameter | float
    LA_RERmax: Parameter | float
    g: Parameter | float
    alpha: Parameter | float
    SCA: Parameter | float
    k: Parameter | float
    Ccont: Parameter | float
    LA_Ccost: Parameter | float
    remob_CStor: Parameter | float
    QN_init: Parameter | float
    RT_ratio_init: Parameter | float
    LA_init: Parameter | float
    DW_init: Parameter | float


class ParamsModel(_ParamsModelBase):
    """ "Instance for a individual, a.k.a. with floats."""

    pass


class ParamsModelDef(_ParamsModelBase):
    """Instance for a definition, a.k.a. with Parameters (object Parameter)"""

    def get_params_model_indiv(
        self,
        estimation_description,
        param,
        latent_variate,
    ) -> ParamsModel:
        """_summary_

        Args:
            param : pc.reals1d_to_params(theta) (NamedTuple)
            estimation_description : NamedTuple(pop, indiv, residuals)
            latent_variate (jnp.array): individual latent variate shape = (estimation_description.nindiv, )

        Returns:
            A ParamsModel object with individual values given by the current latent_variates
        """

        values = {field: getattr(self, field).default_value for field in self._fields}
        # Object._fields returns a dictionary with every attribute of a dataclass or NamedTuple
        # Création d'un dictionnaire (mutable) values, qui pourra être modifié 
        for i, pop_param in enumerate(estimation_description.population_model_parameters):
            values[pop_param] = getattr(
                self, pop_param
            ).parametrization.reals1d_to_params(param.pop[i])

        transformed_latent_variate = (
            param.indiv.loc + param.indiv.scale @ latent_variate
        )
        for i, indiv_param in enumerate(estimation_description.indiv_model_parameters):
            values[indiv_param] = getattr(
                self, indiv_param
            ).parametrization.reals1d_to_params(transformed_latent_variate[i])

        return ParamsModel(**values)  # destructuration du dictionnaire values pour créer une nouvelle instance de ParamsModel
    

@dataclass
class State:
    """
    Definition of a state of the model. 2 class methods to initiate the model and to
    return next state
    """

    dRDW: float  # daily growth in root dry weight
    TDW: float  # total dry weight
    TNQ: float  # total nitrogen
    RDW: float  # root dry weight
    SDW: float  # shoot dry weight
    LMA: float  # specific mass
    R_QCavail: float  # quantity of carbon available after leaf construction
    dRDW_max: float  # maximum daily root growth
    PLA: float  # projected leaf area
    LA: float  # leaf area
    SNQ: float  # nitrogen quantity in shoot
    LNA: float  # specific mass in N
    SR_ratio: float  # shoot/root ratio
    RT_ratio: float  # ratio RDW/TDW
    NUtE: float  # TDW/TNQ
    dLA_N: float  # daily growth in leaf area allowed by available nitrogen
    dLA: float  # daily growth in leaf area
    dLA_max: float  # maximum daily growth in leaf area
    QNuptk: float  # nitrogen uptake on day t
    QNavail: (
        float  # nitrogen available at day t (nitrogen uptake plus nitrogen storage)
    )
    QNStor: float  # quantity of nitrogen in the storage compartment
    dQNStor_in: float  # daily quantity of nitrogen going into the storage compartment
    QCavail: float  # quantity of carbon available (from photosynthesis and storage)
    # QCStor : float # quantity of carbon in the storage compartment)
    # dQCStor_in : float  # daily quantity of carbon going into the storage compartment
    QCprod: float  # quantity of carbon produced by photosynthesis

    # from ParamsModel optenir params params_model

    @classmethod
    def from_init(cls, params_model: ParamsModel):

        dLA = 0
        TDW = params_model.DW_init
        RDW = params_model.RT_ratio_init * TDW
        SDW = TDW - RDW
        LA = (
            params_model.LA_init
        )  # (1 - self.p["RT_ratio_init"]) * TDW * self.p["Ccont"] / self.p["LA_Ccost"]
        SR_ratio = SDW / RDW
        RT_ratio = RDW / TDW
        LMA = SDW / LA
        TNQ = params_model.QN_init
        NUtE = TDW / TNQ
        QNuptk = params_model.SNU * RDW
        QNStor = 0
        QNavail = QNuptk + params_model.remob_NStor * QNStor
        PLA = jnp.minimum(LA, params_model.g * (1 - jnp.exp(-params_model.k * LA)))
        QCprod = params_model.SCA * PLA
        QCavail = QCprod  # + self.p["remob_CStor"] * QCStor
        R_QCavail = jnp.maximum(0, QCavail - dLA * params_model.LA_Ccost)
        dRDW_max = params_model.R_RGRmax * RDW
        dRDW = jnp.minimum(
            jnp.minimum(
                QNavail / params_model.R_NCont_min, R_QCavail / params_model.Ccont
            ),
            dRDW_max,
        )
        dLA_N = params_model.alpha * (
            QNavail
            - jnp.minimum(
                QNavail, R_QCavail * params_model.R_NCont_min / params_model.Ccont
            )
        )
        SNQ = dLA_N
        LNA = SNQ / LA
        dLA_max = params_model.LA_RERmax * LA
        dLA = jnp.maximum(
            0, jnp.minimum(jnp.minimum(QCprod / params_model.LA_Ccost, dLA_N), dLA_max)
        )
        dQNStor_in = jnp.maximum(
            0, QNavail - dRDW * params_model.R_NCont_min - dLA / params_model.alpha
        )

        return cls(
            TDW=TDW,
            RDW=RDW,
            SDW=SDW,
            SR_ratio=SR_ratio,
            RT_ratio=RT_ratio,
            LMA=LMA,
            TNQ=TNQ,
            NUtE=NUtE,
            QNuptk=QNuptk,
            QNStor=QNStor,
            LA=LA,
            PLA=PLA,
            QCprod=QCprod,
            QNavail=QNavail,
            QCavail=QCavail,
            R_QCavail=R_QCavail,
            dRDW_max=dRDW_max,
            dRDW=dRDW,
            dLA_N=dLA_N,
            SNQ=SNQ,
            LNA=LNA,
            dLA_max=dLA_max,
            dLA=dLA,
            dQNStor_in=dQNStor_in,
        )

    def next_state(self, params_model: ParamsModel):

        RDW = self.RDW + self.dRDW
        LA = self.LA + self.dLA
        QNStor = self.QNStor + self.dQNStor_in - params_model.remob_NStor * self.QNStor
        # QCStor = self.QCStor + self.dQCStor_in
        TNQ = self.TNQ + self.QNuptk
        TDW = self.TDW + self.QCprod / params_model.Ccont
        SDW = TDW - RDW
        SNQ = self.SNQ + self.dLA_N
        LNA = SNQ / LA
        SR_ratio = SDW / RDW
        RT_ratio = RDW / TDW
        NUtE = TDW / TNQ
        LMA = SDW / LA
        QNuptk = params_model.SNU * RDW
        QNavail = QNuptk + params_model.remob_NStor * QNStor
        PLA = jnp.minimum(params_model.g * (1 - (jnp.exp(-params_model.k * LA))), LA)
        QCprod = params_model.SCA * PLA
        QCavail = QCprod  # + params_model["remob_CStor"] * QCStor
        R_QCavail = jnp.maximum(0, QCavail - params_model.LA_Ccost * self.dLA)
        dRDW_max = params_model.R_RGRmax * RDW
        dRDW = jnp.minimum(
            jnp.minimum(
                QNavail / params_model.R_NCont_min, R_QCavail / params_model.Ccont
            ),
            dRDW_max,
        )
        dLA_N = params_model.alpha * (
            QNavail
            - jnp.minimum(
                QNavail, R_QCavail * params_model.R_NCont_min / params_model.Ccont
            )
        )
        dLA_max = params_model.LA_RERmax * LA
        dLA = jnp.minimum(jnp.minimum(QCprod / params_model.LA_Ccost, dLA_N), dLA_max)
        dQNStor_in = jnp.maximum(0, (dLA_N - dLA) / params_model.alpha)

        return State(
            TDW=TDW,
            RDW=RDW,
            SDW=SDW,
            SR_ratio=SR_ratio,
            RT_ratio=RT_ratio,
            LMA=LMA,
            TNQ=TNQ,
            NUtE=NUtE,
            QNuptk=QNuptk,
            QNStor=QNStor,
            PLA=PLA,
            QCprod=QCprod,
            QNavail=QNavail,
            QCavail=QCavail,
            R_QCavail=R_QCavail,
            dRDW_max=dRDW_max,
            dRDW=dRDW,
            dLA_N=dLA_N,
            SNQ=SNQ,
            LNA=LNA,
            dLA_max=dLA_max,
            dLA=dLA,
            dQNStor_in=dQNStor_in,
            LA=LA,
        )


def run(params_model: ParamsModel, n_days):
    sta = State.from_init(params_model)
    states = [sta]

    for _ in range(n_days):
        sta = sta.next_state(params_model)
        states.append(sta)

    return states
