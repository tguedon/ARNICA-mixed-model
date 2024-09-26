# ====================================================
# Script Name: mixed_model.py
# Description: Defines the functions used for inference of the statistical parameters of the nonlinear mixed effects model
# with the plant growth model ARNICA.
# 
# Author: Tom Guédon
# Date: 2024-09
# Version: 1.0
#
# Python Version: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
# Système d'exploitation: Windows 10
#
# Contact : tom.guedondurieu@gmail.com
# ====================================================


# pylint: disable= C0116, C0103, R0913, R0914, W0640, E0602
import collections
import typing
from typing import NamedTuple
import jax
import jax.numpy as jnp
from tqdm import tqdm
import parametrization_cookbook.jax as pc
from model import ParamsModelDef, run, Parameter


def list_of_states_to_output(list_of_states):
    return {
        output.name: jnp.array(
            [getattr(list_of_states[t], output.name) for t in output.times]
        )
        for output in estimation_description.outputs
    }


@jax.jit
def true_log_lik_indiv(
    param,
    latent_variate,
    y,
):
    """compute individual log complete likelihood

    Args:
        params_model (Object ParamsModel): default values (fix) + initial values (to estim)
        p_model (jnp.array, shape = (p,)): fixed values of params to estimate
        sd_model (jnp.array, shape = (p,p)): scale parameter of the random effects
        sd (jnp.array, shape = (q,q)): covariance matrix of the responses
        z (jnp.array, shape = (p,)): realisation of random effect
        y (jnp.array, shape = (n_days, q)): individual response column are liste_variable
        params_to_estim (list len = p): names (str) of parameters to estim, order of p_model and sd_model
        liste_variable (list len = q): names (str) of observed variables, order of y (columns)

    Returns:
        float : individual log complete likelihood
    """

    assert len(latent_variate.shape) == 1

    params_model_indiv = base_params_model.get_params_model_indiv(
        estimation_description, param, latent_variate
    )
    list_of_state_pred = run(
        params_model=params_model_indiv, n_days=estimation_description.ndays_max
    )

    y_pred = list_of_states_to_output(list_of_state_pred)

    log_likli_obs = -0.5 * sum(
        (jnp.log(2 * jnp.pi) + jnp.log(param.residual_var[i])) * len(output.times)
        for i, output in enumerate(estimation_description.outputs)
    ) - 0.5 * sum(
        jnp.nansum((jnp.log(y[output.name]) - jnp.log(y_pred[output.name])) ** 2) / param.residual_var[i]
        for i, output in enumerate(estimation_description.outputs)
    )

    log_likli_lat = (
        -estimation_description.nindiv / 2 * jnp.log(2 * jnp.pi)
        - 0.5 * latent_variate @ latent_variate
    )

    return log_likli_lat + log_likli_obs


def log_lik_indiv(
    theta,
    latent_variate,
    y,
):
    """compute individual log complete likelihood

    Args:
        TODO

    Returns:
        float : individual log complete likelihood
    """
    param = parametrization.reals1d_to_params(theta)

    return true_log_lik_indiv(
        param,
        latent_variate,
        y,
    )


grad_indiv = jax.grad(log_lik_indiv, 0)


def log_lik_rows(
    theta,
    latent_variates_all,
    y_all,
):
    """compute array of individuals log complete likelihoods

    Args:
        theta (jnp.array, shape = (parametrization.size,) ): real valued (float) unconstrained parameter
        params_model (Object ParamsModel): default values (fix) + initial values (to estim)
        z_tot (jnp.array, shape = (nindiv,p)): nindiv realisations of random effects
        y_tot (jnp.array, shape = (nindiv, n_days, q)): responses, columns (dim 2) are liste_variable
        params_to_estim (list len = p): names (str) of parameters to estim, order of p_model and sd_model
        liste_variable (list len = q): names (str) of observed variables, order of y (columns)
        parametrization (parametrization_cookbook.namedTuple): parametrization_cook parametrization

    Returns:
        jnp.array, shape = (nindiv,): array of individuals log complete likelihoods
    """

    return jnp.array(
        [
            log_lik_indiv(
                theta,
                latent_variate,
                y,
            )
            for y, latent_variate in zip(y_all, latent_variates_all)
        ]
    )


jac_log_lik_rows = jax.jacfwd(log_lik_rows, 0)


grad_log_lik_indiv = jax.grad(log_lik_indiv, 0)


def jac_log_lik(
    theta,
    latent_variates_all,
    y_all,
):

    return jnp.array(
        [
            grad_log_lik_indiv(
                theta,
                latent_variate,
                y,
            )
            for y, latent_variate in zip(y_all, latent_variates_all)
        ]
    )


def log_lik(
    theta,
    latent_variates_all,
    y_all,
):
    """compute log complete likelihood

    Args:
        theta (jnp.array, shape = (parametrization.size,) ): real valued (float) unconstrained parameter
        params_model (Object ParamsModel): default values (fix) + initial values (to estim)
        z_tot (jnp.array, shape = (nindiv,p)): nindiv realisations of random effects
        y_tot (jnp.array, shape = (nindiv, n_days, q)): responses, columns (dim 2) are liste_variable
        params_to_estim (list len = p): names (str) of parameters to estim, order of p_model and sd_model
        liste_variable (list len = q): names (str) of observed variables, order of y (columns)
        parametrization (parametrization_cookbook.namedTuple): parametrization_cook parametrization

    Returns:
        float: log complete likelihood
    """
    return log_lik_rows(
        theta,
        latent_variates_all,
        y_all,
    ).sum()


def mh_step_gibbs_indiv(
    theta,
    latent_variates_all,
    y_all,
    sigma_proposal,
    prng_key,
):
    """_summary_

    Args:
        theta (jnp.array, shape = (parametrization.size,) ): real valued (float) unconstrained parameter
        params_model (Object ParamsModel): default values (fix) + initial values (to estim)
        z_tot (jnp.array, shape = (nindiv,p)): nindiv realisations of random effects
        y_tot (jnp.array, shape = (nindiv, n_days, q)): responses, columns (dim 2) are liste_variable
        params_to_estim (list len = p): names (str) of parameters to estim, order of p_model and sd_model
        liste_variable (list len = q): names (str) of observed variables, order of y (columns)
        parametrization (parametrization_cookbook.namedTuple): parametrization_cook parametrization
        sigma_proposal (jnp.array, shape=(nombre_indiv, p)): proposal variances of the random effects
        prng_key (jax PRNGkey): seed

    Returns:
        acc (jnp.array, shape=(p,)) : array of acceptations
        z_new (jnp.array, shape=(nindiv,p)) : new random effects
    """
    log_likli = log_lik_rows(
        theta,
        latent_variates_all,
        y_all,
    )
    n, p = latent_variates_all.shape
    acc = jnp.zeros((n, p))
    z_propo = latent_variates_all
    for j, z in enumerate(latent_variates_all.T):
        prng_key, key1, key2 = jax.random.split(prng_key, 3)
        z_propo = z_propo.at[:, j].add(
            jax.random.normal(key1, shape=(n,)) * sigma_proposal.T[j]
        )
        log_likli_propo = log_lik_rows(
            theta,
            z_propo,
            y_all,
        )

        mask = log_likli_propo - log_likli > jnp.log(
            jax.random.uniform(key=key2, shape=(n,))
        )
        z_propo = z_propo.at[:, j].set(jnp.where(mask, z_propo[:, j], z))
        log_likli = jnp.where(mask, log_likli_propo, log_likli)
        acc = acc.at[:, j].set(mask)

    return (acc, z_propo, log_likli)


def mh_step_adaptative_indiv(
    theta,
    latent_variates_all,
    y_all,
    sigma_proposal,
    current_acc,
    step,
    prng_key,
):
    """_summary_

    Args:
        theta (jnp.array, shape = (parametrization.size,) ): real valued (float) unconstrained parameter
        params_model (Object ParamsModel): default values (fix) + initial values (to estim)
        z_tot (jnp.array, shape = (nindiv,p)): nindiv realisations of random effects
        y_tot (jnp.array, shape = (nindiv, n_days, q)): responses, columns (dim 2) are liste_variable
        params_to_estim (list len = p): names (str) of parameters to estim, order of p_model and sd_model
        liste_variable (list len = q): names (str) of observed variables, order of y (columns)
        parametrization (parametrization_cookbook.namedTuple): parametrization_cook parametrization
        sigma_proposal (jnp.array, shape=(p,)): proposal variances of the random effects
        current_acc (jnp.array, shape=(nombre_indiv, p)): acceptance rate for each components of random effects
        step (int): current step in iterations
        prng_key (jax PRNGkey): seed
    Returns:
        sigma_proposal (jnp.array, shape=(p,)) : updated proposal variances of the random effects
        current_acc (jnp.array, shape=(p,)) : updated acceptance rate
        z  (jnp.array, shape=(nindiv,p)) : new random effects
    """

    delta = jnp.exp(min(0.01, (step + 1) ** (-0.5)))

    accept, z, ll = mh_step_gibbs_indiv(
        theta,
        latent_variates_all,
        y_all,
        sigma_proposal,
        prng_key,
    )
    current_acc += 0.02 * (accept - current_acc)
    mask = current_acc > 0.4

    sigma_proposal = jnp.where(mask, sigma_proposal * delta, sigma_proposal / delta)

    return (sigma_proposal, current_acc, z, ll)


class ResSGD(NamedTuple):

    z: jnp.ndarray
    theta: jnp.ndarray
    theta_step: jnp.ndarray
    fim: jnp.ndarray
    key: jnp.ndarray


class ResSARIS(NamedTuple):

    z1: jnp.ndarray
    z0: jnp.ndarray
    theta: jnp.ndarray
    theta0: jnp.array


def one_iter(
        theta,
        z,
        y,
        jac,
        sigma_proposal,
        current_acc,
        step,
        end_heat_step,
        beta,
        prng_key,
):
    
    key, prng_key = jax.random.split(prng_key)

    n, _ = z.shape

    (sigma_proposal, current_acc, z, ll) = mh_step_adaptative_indiv(
        theta,
        z,
        y,
        sigma_proposal,
        current_acc,
        step,
        key,
    )

    current_jac = jac_log_lik_rows(
        theta,
        z,
        y,
    )

    if step < end_heat_step:
        step_val = beta
    else:
        step_val = beta * (step - end_heat_step + 1) ** (-2 / 3)

    jac += step_val * (
        current_jac - jac
    ) 

    grad = jnp.nanmean(jac, axis=0)

    fisher_info_mat = jac.T @ jac / n
    fisher_info_mat = (1 - 0.1 * step_val) * fisher_info_mat + 0.1 * step_val * jnp.eye(parametrization.size)

    theta_step = jnp.linalg.solve(fisher_info_mat, grad)

    theta += step_val * theta_step

    # faire juste tuple et le mettre dans estim ça pour l'attribution
    return theta, z, theta_step, jac, fisher_info_mat, sigma_proposal, current_acc


def estim_param(
        theta_init,
        y,
        seed,
        N_max=1000,
        beta=0.9,
        end_heat_step=300,
        pre_heat=100,
):
    
    prng_key = jax.random.PRNGKey(seed)

    # potentiellement un truc à faire si p=1
    n = len(y)
    p = estimation_description.nindiv

    z = jnp.zeros((n, p))

    res = []

    # mcmc parameters
    sigma_proposal = jnp.ones((n, p))
    current_acc = jnp.ones((n, p)) * 0.4
    theta = theta_init
    jac = jnp.zeros((n, parametrization.size))

    print("heating phase :")
    for step in tqdm(range(pre_heat)):
        key, prng_key = jax.random.split(prng_key)
        (sigma_proposal, current_acc, z, ll) = mh_step_adaptative_indiv(
            theta,
            z,
            y,
            sigma_proposal,
            current_acc,
            step,
            key,
        )
    print("estimation :")
    for step in tqdm(range(N_max)): 

        key, prng_key = jax.random.split(prng_key)
        theta, z, theta_step, jac, fisher_info_mat, sigma_proposal, current_acc = one_iter(
            theta,
            z,
            y,
            jac,
            sigma_proposal,
            current_acc,
            step,
            end_heat_step,
            beta,
            key,
        )

        res.append(ResSGD(z=z, theta=theta, theta_step=theta_step, fim=fisher_info_mat, key=seed))

    return (ResSGD(z=jnp.array([r.z for r in res]),
                   theta=jnp.array([r.theta for r in res]),
                   theta_step=jnp.array([r.theta_step for r in res]),
                   fim=jnp.array([r.fim for r in res]),
                   key=jnp.array([r.key for r in res]),))


# def estim(Y, N_smooth=10000, prng_key=0, pre_heat=1000, end_heat=1500):
#     usefull_values = list(
#         itertools.islice(
#             gsto(
#                 Y,
#                 prng_key=prng_key,
#                 pre_heat=pre_heat,
#                 end_heat=end_heat,
#             ),
#             N_smooth,
#         )
#     )
#     return ResEstim(
#         jnp.array([x.theta for x in usefull_values]).mean(axis=0),
#         jnp.array([x.theta for x in usefull_values]),
#         jnp.array([x.fisher_info_mat for x in usefull_values]).mean(axis=0),
#         jnp.array([x.fisher_info_mat for x in usefull_values]),
#         jnp.array([x.step_mean for x in usefull_values]),
#         jnp.array([x.grad_mean for x in usefull_values]),
#     )


Retdata = collections.namedtuple(
    "Retdata",
    (
        "end_heating",
        "z",
        "theta",
        "fisher_info_mat",
        "step_mean",
        "grad_mean",
    ),
)


ResEstim = collections.namedtuple(
    "ResEstim",
    ("theta", "evol_theta", "fisher_info_mat", "evol_fim", "step_mean", "grad_mean"),
)


# note on typing :
# cette bibliothèque sert a définir les types pour que ce soit clair
# en créant des NamedTuple on doit spécifier les types qu'on attend (pareil dataclasses)
class EstimationDescription(NamedTuple):
    """
    NamedTuple Object that describes the estimation process.
    Attributes :
        indiv_model_parameters : tuple of string that describes the aprameter to consider variable between indivudals
        population_model_parameters : tuple of string that describes the population parameters
        outputs
    """

    indiv_model_parameters: tuple
    population_model_parameters: tuple
    outputs: tuple

    @property
    def nindiv(self):
        return len(self.indiv_model_parameters)

    @property
    def npop(self):
        return len(self.population_model_parameters)

    @property
    def noutput(self):
        return len(self.outputs)

    @property
    def ndays_max(self):
        return max(max(out.times) for out in self.outputs)


class Output(NamedTuple):
    """
    Object taht describes an output.
    Hypothesis : for all outputs, the sequence of point observations is the same for every indidual
                might they be NaN
    """

    name: str
    times: typing.Tuple[int]


def real_output_by_indiv(real_output):
    """
    From a real data of the form :
    real_output = {
        "dRDW": jnp.array(
            [
                [54, 12, 12, 23, 34, 12],
                [54, 12, 12, 23, 34, 13],
                [54, 12, 12, NaN, 34, 14],
            ]
        )  # shape n_total_indiv, n_times_for_this_output
    }
    This function returns a list of dictionnaries of the form :
    [{output_i: seq_time_ouput_i for output in output_list} for i in range(n_indiv)]
    """
    # all outputs must have the same line number
    n_indivs = [x.shape[0] for x in real_output.values()]
    assert all(x == n_indivs[0] for x in n_indivs)
    n_indiv = n_indivs[0]
    return [
        {k: v[i] for k, v in real_output.items()} for i in range(n_indiv)
    ]  # .items: iterer sur (clef,valeur)s


def real_output_all(real_outpout_byindiv):
    """
    from data of the form
        [{output_i: seq_time_ouput_i for output in output_list} for i in range(n_indiv)]
    it returns data of the form of the real data :
    real_output = {
        "dRDW": jnp.array(
            [
                [54, 12, 12, 23, 34, 12],
                [54, 12, 12, 23, 34, 13],
                [54, 12, 12, NaN, 34, 14],
            ]
        )  # shape n_total_indiv, n_times_for_this_output
    }
    """

    return {
        output: jnp.array(
            [real_outpout_byindiv[i][output] for i in range(len(real_outpout_byindiv))]
        )
        for output in real_outpout_byindiv[0].keys()
    }


class DataSimulator:
    """
    a definir
    """

    def __init__(self, nombre_indiv, estimation_description):
        self.nombre_indiv = nombre_indiv
        self.estimation_description = estimation_description

    def get_data(
        self, base_param_model, param, latent_variate_all, prng_key, by_indiv=True
    ):
        """
        base_param_model : ParamModeldef
        param : parametrization.reals1D_to_param
        given a parameter and a latent variate
        return a data_set
        by_indiv type or output type
        prng_key to generate noise
        """

        noise_out = {}
        for k, out in enumerate(self.estimation_description.outputs):
            key, prng_key = jax.random.split(prng_key)
            noise = jnp.sqrt(param.residual_var[k]) * jax.random.normal(
                key, shape=(self.nombre_indiv, len(out.times))
            )
            noise_out[out.name] = noise

        output = []
        for i in range(self.nombre_indiv):

            params_model_indiv = base_param_model.get_params_model_indiv(
                self.estimation_description, param, latent_variate_all[i]
            )

            list_of_state = run(
                params_model=params_model_indiv,
                n_days=self.estimation_description.ndays_max,
            )
            new_out = list_of_states_to_output(list_of_state)

            output.append(
                {
                    out_name: out_val * jnp.exp(noise_out[out_name][i])
                    for out_name, out_val in new_out.items()
                }
            )

        if by_indiv:
            return output
        else:
            return real_output_all(output)


Retdata = collections.namedtuple(
    "Retdata",
    ("z", "theta", "fisher_info_mat", "step_mean", "grad_mean", "comp_log_lik"),
)


ResMd = collections.namedtuple(
    "ResMd",
    ("prop", "acc", "z"),
)


def IS_estimation(theta, y_all, n_is_sample, n_param_sample, seed=1, sd_post_init=None, mean_post_init=None):

    prng_key = jax.random.PRNGKey(seed)
    nombre_indiv = len(y_all)
    p = estimation_description.nindiv
    key, prng_key = jax.random.split(prng_key)
    echantillon_std = jax.random.normal(key=key, shape=(n_is_sample, nombre_indiv, p))

    sd_post = sd_post_init
    mean_post = mean_post_init

    if sd_post_init is None:
        sigma_proposal = jnp.ones((nombre_indiv, p))
        current_acc = jnp.ones((nombre_indiv, p)) * 0.4
        latent_variates_all = jnp.zeros((nombre_indiv, p))
        step = 0
        _, prng_key = jax.random.split(prng_key)

        res_sample = []

        #    print("simulation des echantillons pour calcul des paramètres de distribution d'importance")
        for step in tqdm(range(n_param_sample)):
            _, prng_key = jax.random.split(prng_key)
            (sigma_proposal, current_acc, latent_variates_all, u) = (
                mh_step_adaptative_indiv(
                    theta,
                    latent_variates_all,
                    y_all,
                    sigma_proposal,
                    current_acc,
                    step,
                    prng_key,
                )
            )
            res_sample.append(latent_variates_all)

        sample_mcmc = jnp.array(res_sample)
        sd_post = jnp.std(sample_mcmc[100:, :, :], axis=0)
        mean_post = jnp.mean(sample_mcmc[100:, :, :], axis=0)
    
    if p == 1:
        sd_post = sd_post.reshape(
            nombre_indiv,
        )
        mean_post = mean_post.reshape(
            nombre_indiv,
        )

        echantillon_IS = jnp.array(
            [
                mean_post
                + sd_post
                * ech.reshape(
                    nombre_indiv,
                )
                for ech in echantillon_std
            ]
        )

        #    print("calcul densité gaussienne")
        densite_IS = jnp.array(
            [
                log_multivariate_density_diag(
                    ech.reshape(nombre_indiv, 1), mean_post, sd_post
                )
                for ech in echantillon_IS
            ]
        )
        #    print("calcul des log vraisemblances sur les nouveaux echantillons gaussiens")
        ll_rows_IS = jnp.array(
            [
                log_lik_rows(theta, ech.reshape(nombre_indiv, 1), y_all)
                for ech in tqdm(echantillon_IS)
            ]
        )

    else:
        echantillon_IS = jnp.array(
            [mean_post + sd_post * ech for ech in echantillon_std]
        )
        #    print("calcul densité gaussienne")
        densite_IS = jnp.array(
            [
                log_multivariate_density_diag(ech, mean_post, sd_post)
                for ech in echantillon_IS
            ]
        )
        #    print("calcul des log vraisemblances sur les nouveaux echantillons gaussiens")
        ll_rows_IS = jnp.array(
            [log_lik_rows(theta, ech, y_all) for ech in echantillon_IS]
        )

    res_lik = jnp.exp(ll_rows_IS - densite_IS)
    res = jnp.log(res_lik.mean(axis=0))
    
    if sd_post_init is None:
        return mean_post, sd_post, res

    return jnp.log(res_lik.mean(axis=0))


def log_multivariate_density_diag(sample, mean, std_diag):
    """
    Calcule la densité gaussienne multivariée avec une matrice de covariance diagonale.

    Args:
        sample (jax.numpy.ndarray): échantillon de données, de forme (K, n, p).
        mean (jax.numpy.ndarray): vecteur de moyenne, de forme (n, p).
        std_diag (jax.numpy.ndarray): écart-type des diagonales de la covariance, de forme (n, p).

    Returns:
        jax.numpy.ndarray: vecteur de log densité de la gaussienne multivariée pour chaque n, de forme (n,).
    """

    n, p = sample.shape
    # Calcul de la dimension de l'espace
    if p == 1:
        p = 1
        sample = sample.reshape(n, p)
        mean = mean.reshape(n, p)
        std_diag = std_diag.reshape(n, p)

    # Calcul de la log densité pour chaque échantillon
    diff = sample - mean  # Différence entre l'échantillon et la moyenne
    log_var = 2 * jnp.log(std_diag)  # log de la variance (écart-type au carré)

    # Calcul de la log densité pour chaque dimension indépendamment
    log_probs = -0.5 * ((diff**2) / (std_diag**2))

    # Somme des log densités sur toutes les dimensions pour chaque échantillon
    log_density = jnp.sum(log_probs, axis=1)

    # Ajout du terme de normalisation
    normalization = -0.5 * (p * jnp.log(2 * jnp.pi) - jnp.sum(log_var, axis=1))

    return log_density + normalization


def log_gaussian_rows(y, mean_is, sd_is):
    # Calcul de la partie log(2 * pi * sigma^2)
    log_term = jnp.log(2 * jnp.pi * sd_is ** 2)
    
    # Calcul de la partie (y - mu)^2 / (2 * sigma^2)
    squared_diff = (y - mean_is) ** 2 / (2 * sd_is ** 2)
    
    # Somme des termes pour chaque ligne
    log_density = -0.5 * (log_term + squared_diff).sum(axis=1)
    
    return log_density


base_params_model = ParamsModelDef(
    SNU=Parameter(0.065, pc.RealPositive(scale=0.1)),
    remob_NStor=Parameter(0.037, pc.RealPositive(scale=0.1)),
    remob_CStor=Parameter(0, pc.RealPositive(scale=0.1)),
    R_RGRmax=Parameter(0.34, pc.RealPositive(scale=0.1)),
    R_NCont_min=Parameter(0.024, pc.RealPositive(scale=0.1)),
    LA_RERmax=Parameter(0.23, pc.RealPositive(scale=0.1)),
    g=Parameter(12.72, pc.RealPositive(scale=1)),
    alpha=Parameter(12.7, pc.RealPositive(scale=1)),
    SCA=Parameter(0.1921, pc.RealPositive(scale=0.1)),
    k=Parameter(0.085, pc.RealPositive(scale=0.1)),
    Ccont=Parameter(0.3856, pc.RealPositive(scale=0.1)),
    LA_Ccost=Parameter(0.54, pc.RealPositive(scale=0.1)),
    DW_init=Parameter(0.1, pc.RealPositive(scale=0.1)),
    QN_init=Parameter(0.007, pc.RealPositive(scale=0.1)),
    LA_init=Parameter(0.10, pc.RealPositive(scale=0.1)),
    RT_ratio_init=Parameter(0.20, pc.RealPositive(scale=0.1)),
)


# base_params_model = ParamsModelDef(
#     SNU=Parameter(0.065, pc.RealBounded(bound_lower=0.02, bound_upper=0.1)),
#     remob_NStor=Parameter(0.037, pc.RealBounded(bound_lower=0.185, bound_upper=0.74)),
#     remob_CStor=Parameter(0, pc.RealPositive(scale=0.1)),
#     R_RGRmax=Parameter(0.34, pc.RealBounded(bound_lower=0.1, bound_upper=0.68)),
#     R_NCont_min=Parameter(0.024, pc.RealBounded(bound_lower=0.012, bound_upper=0.028)),
#     LA_RERmax=Parameter(0.23, pc.RealBounded(bound_lower=0.115, bound_upper=0.46)),
#     g=Parameter(12.72, pc.RealBounded(bound_lower=6.0, bound_upper=25.0)),
#     alpha=Parameter(12.7, pc.RealBounded(bound_lower=6.0, bound_upper=25.0)),
#     SCA=Parameter(0.1921, pc.RealBounded(bound_lower=0.096, bound_upper=0.384)),
#     k=Parameter(0.085, pc.RealBounded(bound_lower=0.04, bound_upper=0.17)),
#     Ccont=Parameter(0.3856, pc.RealBounded(bound_lower=0.35, bound_upper=0.45)),
#     LA_Ccost=Parameter(0.54, pc.RealBounded(bound_lower=0.27, bound_upper=1.08)),
#     DW_init=Parameter(0.1, pc.RealBounded(bound_lower=0.05, bound_upper=0.2)),
#     QN_init=Parameter(0.007, pc.RealBounded(bound_lower=0.0035, bound_upper=0.014)),
#     LA_init=Parameter(0.10, pc.RealBounded(bound_lower=0.09, bound_upper=0.12)),
#     RT_ratio_init=Parameter(0.20, pc.RealBounded(bound_lower=0.1, bound_upper=0.4)),
# )


estimation_description = EstimationDescription(
    indiv_model_parameters=("SNU", "SCA",),
    population_model_parameters=("R_RGRmax", "LA_RERmax",),
    outputs=(
        Output("TDW", tuple([i for i in [10, 17, 21]])),
        Output("RDW", tuple([i for i in [10, 17, 21]])),
        Output("TNQ", tuple([i for i in [10, 17, 21]])),
        Output("QNStor", tuple([i for i in [10, 17, 21]])),
        Output("LA", tuple([i for i in [10, 17, 21]])),
        Output("PLA", tuple([i for i in [10, 17, 21]])),
    ),
)


parametrization = pc.NamedTuple(
    pop=pc.Real(shape=estimation_description.npop),
    indiv=pc.NamedTuple(
        loc=pc.Real(shape=estimation_description.nindiv),
        scale=pc.MatrixDiagPosDef(dim=estimation_description.nindiv, scale=0.1),
    ),
    residual_var=pc.RealPositive(shape=estimation_description.noutput),
)