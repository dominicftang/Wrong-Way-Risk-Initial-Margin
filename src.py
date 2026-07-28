import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata
from functools import partial
from scipy.optimize import brentq
import pandas as pd
import os



# Pricing Functions
def bs_call_price(S, K, T, r, q, sigma):
    if T <= 1e-6: return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def bs_put_price(S, K, T, r, q, sigma):
    if T <= 1e-6: return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * K * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)


def forward_price(S, K, T, r, q, **kwargs):
    return (S * np.exp(-q * T) - K * np.exp(-r * T))


# Underlying Assets
def generate_gbm_paths(S0, r, q, sigma, n_paths, mpor_duration_years, rng, **kwargs):
    drift = (r - q - 0.5 * sigma ** 2) * mpor_duration_years
    diffusion = sigma * np.sqrt(mpor_duration_years)
    return S0 * np.exp(drift + diffusion * rng.standard_normal(n_paths))


def generate_merton_paths(S0, r, sigma, lam, gamma, delta_jump, n_paths, mpor_duration_years, rng, q=0.0, **kwargs):
    kbar = np.exp(gamma + 0.5 * delta_jump ** 2) - 1
    drift = (r - q - lam * kbar - 0.5 * sigma ** 2) * mpor_duration_years
    diffusion_coeff = sigma * np.sqrt(mpor_duration_years)
    Z = rng.standard_normal(n_paths)
    brownian_component = np.exp(drift + diffusion_coeff * Z)
    num_jumps = rng.poisson(lam * mpor_duration_years, n_paths)
    total_jump_factor = np.ones(n_paths)
    for i in range(n_paths):
        if num_jumps[i] > 0:
            jumps_log = rng.normal(gamma, delta_jump, num_jumps[i])
            total_jump_factor[i] = np.exp(np.sum(jumps_log))
    return S0 * brownian_component * total_jump_factor


def generate_heston_paths(S0, z0, r, sigma, kappa, eta, rho, n_paths, mpor_days, rng, q=0.0, **kwargs):
    """ Simulates final stock prices using Heston model. """
    dt = 1 / 252.0
    num_steps = mpor_days
    S = np.zeros((num_steps + 1, n_paths));
    S[0, :] = S0
    z = np.zeros((num_steps + 1, n_paths));
    z[0, :] = z0
    for i in range(num_steps):
        alpha, beta = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        z_max = np.maximum(z[i, :], 0)
        z[i + 1, :] = (z[i, :] + kappa * (1 - z[i, :]) * dt +
                       eta * (rho * alpha + np.sqrt(1 - rho ** 2) * beta) * np.sqrt(z_max) * np.sqrt(dt))
        S[i + 1, :] = S[i, :] * np.exp((r - q - 0.5 * sigma ** 2 * z_max) * dt +
                                       sigma * np.sqrt(z_max) * alpha * np.sqrt(dt))
    return S[-1, :]


# Portfolio Increment
def generate_derivative_increments(underlying_path_generator, pricing_function, **params):
    pricing_args_initial = {
        'S': params['S0'], 'K': params['K'], 'T': params['T_current'], 'r': params['r'],
        'q': params['q'], 'sigma': params['pricing_sigma']
    }
    V_initial = pricing_function(**pricing_args_initial)

    S_final = underlying_path_generator(**params)
    pricing_args_final = pricing_args_initial.copy()
    pricing_args_final['S'] = S_final
    pricing_args_final['T'] = params['T_current'] - params['mpor_duration_years']
    V_final = pricing_function(**pricing_args_final)
    return V_final - V_initial


def build_Z(delta_V):
    ranks = rankdata(delta_V, method="average");
    N = delta_V.size
    u = (ranks - 0.5) / N;
    eps = 0.5 / N
    return norm.ppf(np.clip(u, eps, 1.0 - eps))


def weights(z, rho_wwr, h):
    denom = np.sqrt(max(1.0 - rho_wwr ** 2, 1e-9))
    arg = (norm.ppf(h) + rho_wwr * z) / denom
    return norm.cdf(arg) / h


def expected_exposure(delta_V, rho_wwr, IM, h):
    Z = build_Z(delta_V)
    w = weights(Z, rho_wwr, h)
    exposure = np.maximum(delta_V - IM, 0.0)
    return np.mean(w * exposure)

# Calculate CVA
def calculate_cva(rho_wwr, im_levels, params, delta_v_generator):
    h, R, hazard, r = params['h'], params['R'], params['hazard'], params['r']
    time_grid = np.linspace(1 / 252, params['T_maturity'], 25)
    params['mpor_duration_years'] = params['mpor_days'] / 252.0

    ee_profiles = {lbl: [] for lbl in im_levels}
    for t in time_grid:
        params['T_current'] = params['T_maturity'] - t
        delta_V_t = delta_v_generator(**params)
        current_sigma_dv = np.std(delta_V_t)
        for label, z_score in im_levels.items():
            IM = 0.0 if z_score == 0.0 else z_score * current_sigma_dv
            EE_t = expected_exposure(delta_V_t, rho_wwr, IM, h)
            ee_profiles[label].append(EE_t)

    dPD = np.diff(np.insert(1 - np.exp(-hazard * time_grid), 0, 0.0))
    disc = np.exp(-r * time_grid)
    cva_results = {}
    for label, ee_series in ee_profiles.items():
        cva_results[label] = (1 - R) * np.dot(np.array(ee_series) * disc, dPD)
    return cva_results


# =============================================================================
# MAIN SCRIPT WITH MODEL AND PORTFOLIO SWITCHES
# =============================================================================
# Table A and B
# =============================================================================
# MAIN SCRIPT TO GENERATE TABLES A and B
# =============================================================================
"""
if __name__ == "__main__":

    # --- 1. CHOOSE YOUR PORTFOLIO AND UNDERLYING MODEL HERE ---
    PORTFOLIO_TYPE = 'Call'  # Options: 'Call', 'Put', 'Forward'
    UNDERLYING_MODEL = 'Heston'  # Options: 'GBM', 'Merton', 'Heston'

    # --- 2. DEFINE PARAMETERS ---
    common_params = {
        'S0': 100.0, 'K': 100.0, 'T_maturity': 1.0, 'r': 0.05, 'q': 0.0,
        'n_paths': 200000, 'mpor_days': 10, 'h': 0.001,
        'rng': np.random.default_rng(0)
    }
    gbm_params = {'sigma': 0.2}
    merton_params = {'sigma': 0.2, 'lam': 0.05, 'gamma': 0.25, 'delta_jump': 0.25}
    heston_params = {'z0': 0.04, 'sigma': 1.0, 'kappa': 3.0, 'eta': 0.3, 'rho': -0.7}

    # --- 3. SELECT FUNCTIONS AND MERGE PARAMETERS ---
    portfolio_map = {'Call': bs_call_price, 'Put': bs_put_price, 'Forward': forward_price}
    model_map = {'GBM': (generate_gbm_paths, gbm_params), 'Merton': (generate_merton_paths, merton_params),
                 'Heston': (generate_heston_paths, heston_params)}

    pricing_function = portfolio_map.get(PORTFOLIO_TYPE)
    underlying_generator, model_specific_params = model_map.get(UNDERLYING_MODEL)
    params = {**common_params, **model_specific_params}

    if UNDERLYING_MODEL == 'Heston':
        params['pricing_sigma'] = np.sqrt(params['z0'])
    else:
        params['pricing_sigma'] = params['sigma']

    params['mpor_duration_years'] = params['mpor_days'] / 252.0
    params['T_current'] = params['T_maturity']

    delta_v_generator = partial(generate_derivative_increments, underlying_generator, pricing_function)

    # --- 4. GENERATE THE DELTA_V DISTRIBUTION ONCE ---
    print(f"Generating delta_V for a {PORTFOLIO_TYPE} using the {UNDERLYING_MODEL} model...")
    delta_v_dist = delta_v_generator(**params)
    delta_v_std_dev = np.std(delta_v_dist)
    print(f"Standard deviation of delta_V is: {delta_v_std_dev:.4f}")

    # --- 5. CALCULATE ABSOLUTE EE FOR A RANGE OF RHO VALUES ---
    im_levels = {'No IM': 0.0, 'IM @ 95%': norm.ppf(0.95), 'IM @ 99%': norm.ppf(0.99)}
    rhos_for_table = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    ee_results_absolute = {lbl: [] for lbl in im_levels}

    print("\nCalculating Expected Exposure for each rho...")
    for rho in rhos_for_table:
        for label, z_score in im_levels.items():
            IM = 0.0 if z_score == 0.0 else z_score * delta_v_std_dev
            ee = expected_exposure(delta_v_dist, rho, IM, params['h'])
            ee_results_absolute[label].append(ee)
    print("Calculation complete.")

    # --- 6. CONSTRUCT AND DISPLAY TABLE A ---
    table_a_data = {}
    for label, ee_values in ee_results_absolute.items():
        baseline_ee = ee_values[0]
        if baseline_ee > 1e-9:  # Avoid division by zero
            ratios = [ee / baseline_ee for ee in ee_values[1:]]
        else:
            ratios = [np.inf] * (len(ee_values) - 1)
        table_a_data[label] = [baseline_ee] + ratios

    df_a = pd.DataFrame.from_dict(table_a_data, orient='index',
                                  columns=['EE(rho=0)', 'rho=0.1', 'rho=0.25', 'rho=0.5', 'rho=0.75', 'rho=1.0'])

    df_a['EE(rho=0)'] = df_a['EE(rho=0)'].map('{:,.4f}'.format)
    for col in df_a.columns[1:]:
        df_a[col] = df_a[col].map('{:,.1f}'.format)

    print("\n--- Table A: The EE subject to leveraged WWR relative to the EE at rho=0 ---")
    print(df_a)

    # --- 7. CONSTRUCT AND DISPLAY TABLE B ---
    table_b_data = {}
    ee_no_im = np.array(ee_results_absolute['No IM'])

    for label in ['IM @ 95%', 'IM @ 99%']:
        ee_with_im = np.array(ee_results_absolute[label])
        # Calculate ratio EE(with IM) / EE(without IM) for each rho
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(ee_no_im > 1e-9, ee_with_im / ee_no_im, np.nan)
        table_b_data[label] = ratios

    df_b = pd.DataFrame.from_dict(table_b_data, orient='index',
                                  columns=['rho=0.0', 'rho=0.1', 'rho=0.25', 'rho=0.5', 'rho=0.75', 'rho=1.0'])

    # Format as percentages
    for col in df_b.columns:
        df_b[col] = df_b[col].map('{:,.1%}'.format)

    print("\n\n--- Table B: The EE of the portfolio with IM coverage relative to the EE without IM ---")
    print(df_b)
"""



# Figure and Tables (Expected Exposure)
if __name__ == "__main__":

    # 1. CHOOSE PORTFOLIO AND UNDERLYING MODEL
    PORTFOLIO_TYPE = 'Call'  # Options: 'Call', 'Put', 'Forward'
    UNDERLYING_MODEL = 'Merton'  # Options: 'GBM', 'Merton', 'Heston'

    # 2. DEFINE PARAMETERS
    common_params = {
        'S0': 100, 'K': 100, 'T_maturity': 1.0, 'r': 0.05, 'q': 0.0,
        'n_paths': 200000, 'mpor_days': 10, 'h': 0.001,
        'rng': np.random.default_rng(0)
    }
    gbm_params = {'sigma': 0.2}
    merton_params = {'sigma': 0.2, 'lam': 0.05, 'gamma': 0.25, 'delta_jump': 0.25}
    heston_params = {'z0': 0.04, 'sigma': 1.0, 'kappa': 3.0, 'eta': 0.3, 'rho': -0.7}

    # 3. FUNCTIONS AND MERGE PARAMETERS
    portfolio_map = {
        'Call': bs_call_price, 'Put': bs_put_price, 'Forward': forward_price
    }
    pricing_function = portfolio_map.get(PORTFOLIO_TYPE)

    model_map = {
        'GBM': (generate_gbm_paths, gbm_params),
        'Merton': (generate_merton_paths, merton_params),
        'Heston': (generate_heston_paths, heston_params)
    }
    underlying_generator, model_specific_params = model_map.get(UNDERLYING_MODEL)

    if not pricing_function or not underlying_generator:
        raise ValueError("Invalid PORTFOLIO_TYPE or UNDERLYING_MODEL selected.")

    params = {**common_params, **model_specific_params}


    if UNDERLYING_MODEL == 'Heston':
        params['pricing_sigma'] = np.sqrt(params['z0'])
    else:
        params['pricing_sigma'] = params['sigma']

    params['mpor_duration_years'] = params['mpor_days'] / 252.0
    # For a single point EE analysis, we can pick a representative time, e.g., t=0
    params['T_current'] = params['T_maturity']

    delta_v_generator = partial(generate_derivative_increments, underlying_generator, pricing_function)

    # 4. GENERATE THE DELTA_V DISTRIBUTION ONCE
    print(f"Generating delta_V for a {PORTFOLIO_TYPE} using the {UNDERLYING_MODEL} model...")
    delta_v_dist = delta_v_generator(**params)
    delta_v_std_dev = np.std(delta_v_dist)
    print(f"Standard deviation of delta_V is: {delta_v_std_dev:.4f}")

    # 5. CALCULATE EE FOR A RANGE OF RHO VALUES
    im_levels = {'No IM': 0.0, 'IM @ 95%': norm.ppf(0.95), 'IM @ 99%': norm.ppf(0.99)}
    rhos_wwr = np.linspace(0, 1.0, 21)
    ee_results_normalized = {lbl: [] for lbl in im_levels}

    print("\nCalculating Expected Exposure for each rho...")
    for rho in rhos_wwr:
        for label, z_score in im_levels.items():
            IM = 0.0 if z_score == 0.0 else z_score * delta_v_std_dev
            ee = expected_exposure(delta_v_dist, rho, IM, params['h'])
            ee_normalized = ee / delta_v_std_dev
            ee_results_normalized[label].append(ee_normalized)
    print("Calculation complete.")

    # 6. PLOT THE RESULTS
    plt.figure(figsize=(8, 5))
    plt.plot(rhos_wwr, ee_results_normalized['No IM'], label='No IM')
    plt.plot(rhos_wwr, ee_results_normalized['IM @ 95%'], label='IM @ 95%')
    plt.plot(rhos_wwr, ee_results_normalized['IM @ 99%'], label='IM @ 99%')

    plt.xlabel(r'Wrong-way correlation $\rho_{WWR}$')
    plt.ylabel('Expected Exposure (in units of st. dev. of $\delta V$)')
    plt.title(f'EE vs. WWR for a {PORTFOLIO_TYPE} (underlying: {UNDERLYING_MODEL})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

"""
# =============================================================================
# CVA Calculations
# =============================================================================
if __name__ == "__main__":

    # 1. SETUP
    PORTFOLIO_TYPE = 'Call'
    UNDERLYING_MODEL = 'Merton'
    BASE_HAZARD_RATE = 0.05

    # Parmeters
    common_params = {
        'S0': 100.0, 'K': 100.0, 'T_maturity': 1.0, 'r': 0.05, 'q': 0.0,
        'n_paths': 100000, 'mpor_days': 10, 'h': 0.001, 'R': 0.4,
        'rng': np.random.default_rng(0)
    }
    gbm_params = {'sigma': 0.2}
    merton_params = {'sigma': 0.2, 'lam': 0.05, 'gamma': 0.25, 'delta_jump': 0.25}
    heston_params = {'z0': 0.04, 'sigma': 1.0, 'kappa': 3.0, 'eta': 0.3, 'rho': -0.7}

    # Model and combine parameters
    portfolio_map = {'Call': bs_call_price, 'Put': bs_put_price, 'Forward': forward_price}
    model_map = {'GBM': (generate_gbm_paths, gbm_params), 'Merton': (generate_merton_paths, merton_params),
                 'Heston': (generate_heston_paths, heston_params)}

    pricing_function = portfolio_map.get(PORTFOLIO_TYPE)
    underlying_generator, model_specific_params = model_map.get(UNDERLYING_MODEL)
    params = {**common_params, **model_specific_params, 'hazard': BASE_HAZARD_RATE}

    if UNDERLYING_MODEL == 'Heston':
        params['pricing_sigma'] = np.sqrt(params['z0'])
    else:
        params['pricing_sigma'] = params['sigma']

    delta_v_generator = partial(generate_derivative_increments, underlying_generator, pricing_function)

    # 2. CALCULATE ORIGINAL CVA VALUES (FIXED LAMBDA)
    im_levels = {'No IM': 0.0, 'IM @ 95%': norm.ppf(0.95), 'IM @ 99%': norm.ppf(0.99)}
    rhos_wwr = np.linspace(0.1, 0.9, 9)

    print("Step 1: Calculating original CVA values with fixed lambda...")
    original_cva_results = {lbl: [] for lbl in im_levels}
    for rho in rhos_wwr:
        cva_out = calculate_cva(rho, im_levels, params, delta_v_generator)
        for lbl in im_levels:
            original_cva_results[lbl].append(cva_out[lbl])

    # 3. ESTABLISH BASELINE CVA*
    cva_star_dict = calculate_cva(0.0, im_levels, params, delta_v_generator)
    CVA_STAR = cva_star_dict['No IM']
    print(f"\nBaseline CVA* (No IM, rho=0, lambda={BASE_HAZARD_RATE}) = {CVA_STAR:.5f}")

    # 4. SOLVE FOR LAMBDA* AND CALCULATE NEW IM CVA VALUES
    print("\nStep 2: Solving for equivalent lambda* for each rho...")

    def cva_solver(lambda_trial, rho_val, target_cva):
        current_params = params.copy()
        current_params['hazard'] = lambda_trial
        cva_out = calculate_cva(rho_val, {'No IM': 0.0}, current_params, delta_v_generator)
        return cva_out['No IM'] - target_cva

    equivalent_lambdas = []
    new_cva_results = {lbl: [] for lbl in ['IM @ 95%', 'IM @ 99%']}

    for rho in rhos_wwr:
        try:
            lambda_star = brentq(cva_solver, 1e-6, 1.0, args=(rho, CVA_STAR))
            equivalent_lambdas.append(lambda_star)
            print(f"  For rho = {rho:.2f}, equivalent lambda* = {lambda_star:.5f}")

            params_new_lambda = params.copy()
            params_new_lambda['hazard'] = lambda_star
            cva_out_new = calculate_cva(rho, im_levels, params_new_lambda, delta_v_generator)
            for lbl in ['IM @ 95%', 'IM @ 99%']:
                new_cva_results[lbl].append(cva_out_new[lbl])

        except ValueError:
            print(f"  Could not find a solution for rho = {rho:.2f}.")
            equivalent_lambdas.append(np.nan)
            for lbl in ['IM @ 95%', 'IM @ 99%']:
                new_cva_results[lbl].append(np.nan)

    # 5. COMPILE AND DISPLAY RESULTS
    print("\n--- Final Comparison ---")
    results_df = pd.DataFrame({
        'rho_WWR': rhos_wwr,
        'Equivalent_Lambda_Star': equivalent_lambdas,
        'Original_CVA_95': original_cva_results['IM @ 95%'],
        'New_CVA_95': new_cva_results['IM @ 95%'],
        'Original_CVA_99': original_cva_results['IM @ 99%'],
        'New_CVA_99': new_cva_results['IM @ 99%']
    })

    results_df['Proportion_95'] = results_df['New_CVA_95'] / results_df['Original_CVA_95']
    results_df['Proportion_99'] = results_df['New_CVA_99'] / results_df['Original_CVA_99']

    pd.set_option('display.float_format', '{:.5f}'.format)
    print(results_df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['rho_WWR'], results_df['Proportion_95'], 'o-', label='IM @ 95% (New CVA / Original CVA)')
    plt.plot(results_df['rho_WWR'], results_df['Proportion_99'], 's-', label='IM @ 99% (New CVA / Original CVA)')
    plt.xlabel(r'Wrong-way correlation $\rho_{WWR}$')
    plt.ylabel('Proportion of Original CVA')
    plt.title('Comparison of CVA with Fixed vs. Equivalent Hazard Rate')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()




"""
if __name__ == "__main__":

    # --- 1. DEFINE ALL SCENARIOS TO RUN ---
    scenarios = [
        # --- CALL OPTIONS ---
        # GBM Scenarios
        {'name': 'GBM_OTM_Call', 'model': 'GBM', 'portfolio': 'Call', 'K': 120},
        {'name': 'GBM_ATM_Call', 'model': 'GBM', 'portfolio': 'Call', 'K': 100},
        {'name': 'GBM_ITM_Call', 'model': 'GBM', 'portfolio': 'Call', 'K': 80},
        {'name': 'GBM_ITM_Put', 'model': 'GBM', 'portfolio': 'Put', 'K': 120},
        {'name': 'GBM_ATM_Put', 'model': 'GBM', 'portfolio': 'Put', 'K': 100},
        {'name': 'GBM_OTM_Put', 'model': 'GBM', 'portfolio': 'Put', 'K': 80},
        {'name': 'GBM_Forward', 'model': 'GBM', 'portfolio': 'Forward', 'K': 100},
        # Heston Scenarios (Negative and Positive Correlation)
        {'name': 'Heston_NegRho_OTM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 120, 'rho': -0.7},
        {'name': 'Heston_NegRho_ATM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 100, 'rho': -0.7},
        {'name': 'Heston_NegRho_ITM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 80, 'rho': -0.7},
        {'name': 'Heston_PosRho_OTM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 120, 'rho': 0.7},
        {'name': 'Heston_PosRho_ATM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 100, 'rho': 0.7},
        {'name': 'Heston_PosRho_ITM_Call', 'model': 'Heston', 'portfolio': 'Call', 'K': 80, 'rho': 0.7},

        {'name': 'Heston_NegRho_ITM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 120, 'rho': -0.7},
        {'name': 'Heston_NegRho_ATM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 100, 'rho': -0.7},
        {'name': 'Heston_NegRho_OTM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 80, 'rho': -0.7},
        {'name': 'Heston_PosRho_ITM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 120, 'rho': 0.7},
        {'name': 'Heston_PosRho_ATM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 100, 'rho': 0.7},
        {'name': 'Heston_PosRho_OTM_Put', 'model': 'Heston', 'portfolio': 'Put', 'K': 80, 'rho': 0.7},

        {'name': 'Heston_PosRho_Forward', 'model': 'Heston', 'portfolio': 'Forward', 'K': 100, 'rho': 0.7},
        {'name': 'Heston_NegRho_Forward', 'model': 'Heston', 'portfolio': 'Forward', 'K': 100, 'rho': -0.7},
        # Merton Scenarios (Positive and Negative Jumps)
        {'name': 'Merton_PosJump_ATM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 100, 'gamma': 0.25},
        {'name': 'Merton_PosJump_OTM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 120, 'gamma': 0.25},
        {'name': 'Merton_PosJump_ITM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 80, 'gamma': 0.25},
        {'name': 'Merton_NegJump_OTM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 120, 'gamma': -0.25},
        {'name': 'Merton_NegJump_ITM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 80, 'gamma': -0.25},
        {'name': 'Merton_NegJump_ATM_Call', 'model': 'Merton', 'portfolio': 'Call', 'K': 100, 'gamma': -0.25},

        {'name': 'Merton_PosJump_ATM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 100, 'gamma': 0.25},
        {'name': 'Merton_PosJump_OTM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 80, 'gamma': 0.25},
        {'name': 'Merton_PosJump_ITM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 120, 'gamma': 0.25},
        {'name': 'Merton_NegJump_OTM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 80, 'gamma': -0.25},
        {'name': 'Merton_NegJump_ITM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 120, 'gamma': -0.25},
        {'name': 'Merton_NegJump_ATM_Put', 'model': 'Merton', 'portfolio': 'Put', 'K': 100, 'gamma': -0.25},

        {'name': 'Merton_PosJump_forward', 'model': 'Merton', 'portfolio': 'Forward', 'K': 100, 'gamma': 0.25},
        {'name': 'Merton_NegJump_forward', 'model': 'Merton', 'portfolio': 'Forward', 'K': 100, 'gamma': -0.25},
    ]


    plot_output_dir = "cva_comparison_plots"
    table_output_dir = "cva_results_tables"
    if not os.path.exists(plot_output_dir): os.makedirs(plot_output_dir)
    if not os.path.exists(table_output_dir): os.makedirs(table_output_dir)

    for scenario in scenarios:
        print(f"\n{'=' * 20} RUNNING SCENARIO: {scenario['name']} {'=' * 20}")

        PORTFOLIO_TYPE = scenario['portfolio']
        UNDERLYING_MODEL = scenario['model']
        BASE_HAZARD_RATE = 0.05
        n_paths_for_run = 100000

        common_params = {
            'S0': 100.0, 'K': scenario['K'], 'T_maturity': 1.0, 'r': 0.05, 'q': 0.0,
            'n_paths': n_paths_for_run, 'mpor_days': 10, 'h': 0.001, 'R': 0.4,
            'rng': np.random.default_rng(0)
        }
        gbm_params = {'sigma': 0.2}
        merton_params = {'sigma': 0.2, 'lam': 0.05, 'gamma': 0.25, 'delta_jump': scenario.get('delta_jump', 0.25)}
        heston_params = {'z0': 0.04, 'sigma': 1.0, 'kappa': 3.0, 'eta': 0.3, 'rho': scenario.get('rho', -0.7)}

        portfolio_map = {'Call': bs_call_price, 'Put': bs_put_price, 'Forward': forward_price}
        model_map = {'GBM': (generate_gbm_paths, gbm_params), 'Merton': (generate_merton_paths, merton_params),
                     'Heston': (generate_heston_paths, heston_params)}

        pricing_function = portfolio_map.get(PORTFOLIO_TYPE)
        underlying_generator, model_specific_params = model_map.get(UNDERLYING_MODEL)
        params = {**common_params, **model_specific_params, 'hazard': BASE_HAZARD_RATE}

        if UNDERLYING_MODEL == 'Heston':
            params['pricing_sigma'] = np.sqrt(params['z0'])
        else:
            params['pricing_sigma'] = params['sigma']

        delta_v_generator = partial(generate_derivative_increments, underlying_generator, pricing_function)

        im_levels = {'No IM': 0.0, 'IM @ 95%': norm.ppf(0.95), 'IM @ 99%': norm.ppf(0.99)}

        # Run Comparison Analysis for Graph
        rhos_for_graph = np.linspace(0.1, 0.9, 9)
        original_cva_results_graph = {lbl: [] for lbl in im_levels}
        for rho in rhos_for_graph:
            cva_out = calculate_cva(rho, im_levels, params, delta_v_generator)
            for lbl in im_levels: original_cva_results_graph[lbl].append(cva_out[lbl])

        cva_star_dict = calculate_cva(0.0, im_levels, params, delta_v_generator)
        CVA_STAR = cva_star_dict['No IM']


        def cva_solver(lambda_trial, rho_val, target_cva):
            current_params = params.copy();
            current_params['hazard'] = lambda_trial
            cva_out = calculate_cva(rho_val, {'No IM': 0.0}, current_params, delta_v_generator)
            return cva_out['No IM'] - target_cva


        equivalent_lambdas = []
        new_cva_results_graph = {lbl: [] for lbl in ['IM @ 95%', 'IM @ 99%']}
        for rho in rhos_for_graph:
            try:
                lambda_star = brentq(cva_solver, 1e-6, 1.0, args=(rho, CVA_STAR))
                equivalent_lambdas.append(lambda_star)
                params_new_lambda = params.copy();
                params_new_lambda['hazard'] = lambda_star
                cva_out_new = calculate_cva(rho, im_levels, params_new_lambda, delta_v_generator)
                for lbl in ['IM @ 95%', 'IM @ 99%']: new_cva_results_graph[lbl].append(cva_out_new[lbl])
            except (ValueError, RuntimeError):
                equivalent_lambdas.append(np.nan);
                for lbl in ['IM @ 95%', 'IM @ 99%']: new_cva_results_graph[lbl].append(np.nan)

        # Compile and Export Comparison Table
        results_df = pd.DataFrame({'rho_WWR': rhos_for_graph, 'Equivalent_Lambda_Star': equivalent_lambdas,
                                   'Original_CVA_95': original_cva_results_graph.get('IM @ 95%', []),
                                   'New_CVA_95': new_cva_results_graph.get('IM @ 95%', []),
                                   'Original_CVA_99': original_cva_results_graph.get('IM @ 99%', []),
                                   'New_CVA_99': new_cva_results_graph.get('IM @ 99%', [])})
        results_df['Proportion_95'] = results_df['New_CVA_95'] / results_df['Original_CVA_95']
        results_df['Proportion_99'] = results_df['New_CVA_99'] / results_df['Original_CVA_99']
        print(f"\n--- Final Comparison Table for {scenario['name']} ---");
        pd.set_option('display.float_format', '{:.5f}'.format);
        print(results_df.dropna())
        table_path = os.path.join(table_output_dir, f"{scenario['name']}_comparison_table.csv");
        results_df.to_csv(table_path)
        print(f"Comparison table saved to {table_path}")

        # Plot Graph
        fig, ax = plt.subplots(figsize=(10, 6));
        ax.plot(results_df['rho_WWR'], results_df['Proportion_95'], 'o-', label='IM @ 95% (New CVA / Original CVA)');
        ax.plot(results_df['rho_WWR'], results_df['Proportion_99'], 's-', label='IM @ 99% (New CVA / Original CVA)');
        ax.set_xlabel(r'Wrong-way correlation $\rho_{WWR}$');
        ax.set_ylabel('Proportion of Original CVA');
        ax.set_title(f"CVA Comparison - {scenario['name']}");
        ax.legend();
        ax.grid(True, linestyle=':');
        ax.set_ylim(bottom=0);
        plt.tight_layout()
        plot_path = os.path.join(plot_output_dir, f"{scenario['name']}.png");
        plt.savefig(plot_path);
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

        # Calculate and Export CVA Amplification Table (Original Lambda)
        rhos_for_table = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        cva_table_results = {lbl: {} for lbl in im_levels}
        for rho in rhos_for_table:
            cva_out = calculate_cva(rho, im_levels, params, delta_v_generator)
            for lbl in im_levels: cva_table_results[lbl][rho] = cva_out[lbl]

        table_a_data = {}
        for label, cva_values in cva_table_results.items():
            baseline_cva = cva_values[0.0];
            if baseline_cva > 1e-9:
                ratios = [cva_values[rho] / baseline_cva for rho in rhos_for_table[1:]]
            else:
                ratios = [np.inf] * (len(rhos_for_table) - 1)
            table_a_data[label] = [baseline_cva] + ratios

        df_a = pd.DataFrame.from_dict(table_a_data, orient='index',
                                      columns=['CVA(rho=0)', 'rho=0.1', 'rho=0.25', 'rho=0.5', 'rho=0.75', 'rho=1.0'])
        print(f"\n--- CVA Amplification Table (Original Lambda) for {scenario['name']} ---");
        df_a_display = df_a.copy();
        df_a_display['CVA(rho=0)'] = df_a_display['CVA(rho=0)'].map('{:,.4f}'.format)
        for col in df_a_display.columns[1:]: df_a_display[col] = df_a_display[col].map('{:,.1f}'.format)
        print(df_a_display)
        table_a_path = os.path.join(table_output_dir, f"{scenario['name']}_amplification_table.csv");
        df_a.to_csv(table_a_path)
        print(f"Amplification table saved to {table_a_path}")

        # Calculate and Export CVA Amplification Table (Equivalent Lambda*)
        lambda_star_table = {}
        for rho in rhos_for_table:
            if rho == 0.0:
                lambda_star_table[rho] = BASE_HAZARD_RATE
            else:
                try:
                    lambda_star_table[rho] = brentq(cva_solver, 1e-6, 1.0, args=(rho, CVA_STAR))
                except (ValueError, RuntimeError):
                    lambda_star_table[rho] = np.nan

        cva_derisked_table_results = {lbl: {} for lbl in im_levels}
        for rho in rhos_for_table:
            params_new_lambda = params.copy();
            params_new_lambda['hazard'] = lambda_star_table[rho]
            cva_out = calculate_cva(rho, im_levels, params_new_lambda, delta_v_generator)
            for lbl in im_levels: cva_derisked_table_results[lbl][rho] = cva_out[lbl]

        table_a_derisked_data = {}
        for label, cva_values in cva_derisked_table_results.items():
            baseline_cva = cva_derisked_table_results[label][0.0]
            if baseline_cva > 1e-9:
                ratios = [cva_values[rho] / baseline_cva for rho in rhos_for_table[1:]]
            else:
                ratios = [np.inf] * (len(rhos_for_table) - 1)
            table_a_derisked_data[label] = [baseline_cva] + ratios

        df_a_derisked = pd.DataFrame.from_dict(table_a_derisked_data, orient='index',
                                               columns=['CVA(rho=0)', 'rho=0.1', 'rho=0.25', 'rho=0.5', 'rho=0.75',
                                                        'rho=1.0'])
        print(f"\n--- De-Risked CVA Amplification Table (Equivalent Lambda*) for {scenario['name']} ---");
        df_a_derisked_display = df_a_derisked.copy();
        df_a_derisked_display['CVA(rho=0)'] = df_a_derisked_display['CVA(rho=0)'].map('{:,.4f}'.format)
        for col in df_a_derisked_display.columns[1:]: df_a_derisked_display[col] = df_a_derisked_display[col].map(
            '{:,.1f}'.format)
        print(df_a_derisked_display)
        table_a_derisked_path = os.path.join(table_output_dir, f"{scenario['name']}_amplification_table_derisked.csv");
        df_a_derisked.to_csv(table_a_derisked_path)
        print(f"De-Risked Amplification table saved to {table_a_derisked_path}")

