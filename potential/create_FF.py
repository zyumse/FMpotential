import numpy as np
from scipy import optimize
import scipy.constants as scc
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Constants and Initialization
DELTA_R = 0.02  
R_MAX = 8 + DELTA_R
KCALPERMOLE2EV = 1000 * scc.calorie / scc.Avogadro / scc.electron_volt
QF_SI = 0.82
QF_O = -QF_SI / 2
SR_CUT = 5.5
GAMMA = 0.2

def load_data(path):
    data = np.load(path)
    return data['f_SiSi3'], data['f_SiO3'], data['f_OO3']

def force_form1(r, qq, A, B, C):
    part1 = (1e20 * qq / r**2 * scc.elementary_charge**2 / 4 / np.pi / scc.epsilon_0 / 1e10 / scc.electron_volt / KCALPERMOLE2EV)
    part2 = (A * B * np.exp(-B * r) / KCALPERMOLE2EV)
    part3 = (-6 * C / r**7 / KCALPERMOLE2EV)
    return part1 + part2 + part3

def F_wolf(r, q1, q2):
    return (1e20 * q1 * q2 * (1 / r**2 - 1 / 8**2) * scc.elementary_charge**2 / 4 / np.pi / scc.epsilon_0 / 1e10 / scc.electron_volt / KCALPERMOLE2EV)

def curve_fit_parameters(R, data, bounds, initial_guess):
    xfit = R[(R > bounds[0]) * (R < SR_CUT)]
    yfit = data[(R > bounds[0]) * (R < SR_CUT)]
    params, _ = optimize.curve_fit(force_form1, xfit, yfit, p0=initial_guess, maxfev=1000000)
    return params

def compute_ffnew(r, rcut, params, q1, q2):
    ffnew = np.zeros((len(r), 3))
    ffnew[:, 0] = r
    
    # Calculate force within SR_CUT and rcut
    mask_sr_cut = (r >= rcut) * (r <= SR_CUT)
    ffnew[mask_sr_cut, 2] = force_form1(r[mask_sr_cut], *params)
    
    # Calculate force beyond SR_CUT using Wolf method
    ffnew[r > SR_CUT, 2] = F_wolf(r[r > SR_CUT], q1, q2)
    
    # Smooth transition
    cut_tmp = ffnew[r > SR_CUT, 2][0]
    ffnew[mask_sr_cut, 2] = (ffnew[mask_sr_cut, 2] - cut_tmp) * np.exp(-GAMMA**2 / (r[mask_sr_cut] - SR_CUT)**2) + cut_tmp

    # Linear extrapolation for R < rcut
    # Determine the slope at rcut
    slope_at_rcut = (force_form1(rcut + 0.01, *params) - force_form1(rcut, *params)) / 0.01
    ffnew[r < rcut, 2] = force_form1(rcut, *params) + slope_at_rcut * (r[r < rcut] - rcut)
    
    # Integrate force to get potential energy
    ffnew[1:, 1] = -cumulative_trapezoid(ffnew[:, 2], x=r)
    ffnew[:, 1] = ffnew[:, 1] - ffnew[-1, 1]
    
    return ffnew

def compute_smoothed_ff(r, R, f, q1, q2, rcut, SR_cut, gamma):
    ffnew = np.zeros((len(r), 3))
    ffnew[:, 0] = r

    # Smoothed force for r in [rcut, SR_cut]
    mask_r = (r >= rcut) * (r <= SR_cut)
    mask_R = (R >= rcut) * (R <= SR_cut)
    ffnew[mask_r, 2] = np.interp(r[mask_r], R[mask_R], savgol_filter(f[mask_R], 101, 5))

    # Wolf force for r > SR_cut
    ffnew[r > SR_cut, 2] = F_wolf(r[r > SR_cut], q1, q2)

    # Smooth transition
    cut_tmp = ffnew[r > SR_cut, 2][0]
    ffnew[mask_r, 2] = (ffnew[mask_r, 2] - cut_tmp) * np.exp(-gamma**2 / (r[mask_r] - SR_cut)**2) + cut_tmp

    # Linear extrapolation for r < rcut
    xfit = ffnew[mask_r,0][:50]
    yfit = ffnew[mask_r,2][:50]
    slope, intercept = np.polyfit(xfit, yfit, 1)
    ffnew[r < rcut, 2] = slope * ffnew[r < rcut, 0] + intercept

    # Integrate force to get potential
    ffnew[1:, 1] = -cumulative_trapezoid(ffnew[:, 2], x=r)
    ffnew[:, 1] = ffnew[:, 1] - ffnew[-1, 1]
    
    return ffnew


def save_for_lammps(filename, data):
    header = f"# LAMMPS tabulated potential\n#\n# Columns: Index R Potential Force"
    np.savetxt(filename, data, fmt=['%d', '%.6f', '%.6f', '%.6f'], header=header, comments='')

def plot_forces(ffnew_SiSi, ffnew_SiO, ffnew_OO, R, f_SiSi3, f_SiO3, f_OO3):
    plt.figure(dpi=200, figsize=[4,3])

    # Plot fitted (FM-fit) forces
    plt.plot(ffnew_SiSi[:,0], ffnew_SiSi[:,2], label='FM-fit SiSi')
    plt.plot(ffnew_SiO[:,0], ffnew_SiO[:,2], label='FM-fit SiO')
    plt.plot(ffnew_OO[:,0], ffnew_OO[:,2], label='FM-fit OO')

    # Plot original (FM-DFT) forces with circles
    plt.plot(R, f_SiSi3, 'C0o', ms=3, label='FM-DFT SiSi')
    plt.plot(R, f_SiO3, 'C1o', ms=3, label='FM-DFT SiO')
    plt.plot(R, f_OO3, 'C2o', ms=3, label='FM-DFT OO')

    # Setting plot limits and legend
    plt.ylim(-800, 1000)
    plt.xlim(0, 6)
    plt.legend(fontsize=8)
    plt.show()

def main(mode):
    # Load data
    f_SiSi3, f_SiO3, f_OO3 = load_data('../data/results.npz')
    r = np.arange(DELTA_R, R_MAX, DELTA_R)
    R = r[1:] - DELTA_R / 2

    r = np.arange(0.001, 8.001, 0.001)
    if mode == 'fit':
        # Fit parameters
        params_SiSi = curve_fit_parameters(R, f_SiSi3, [2.4, 5.5], [QF_SI**2, 1388, 3, 175])
        params_SiO = curve_fit_parameters(R, f_SiO3, [1.3, 5.5], [QF_O * QF_SI, 18000, 4.8, 133])
        params_OO = curve_fit_parameters(R, f_OO3, [2.2, 5.5], [QF_O**2, 1388, 4, 300])
        ffnew_SiO = compute_ffnew(r, 1.3, params_SiO, QF_O, QF_SI)
        ffnew_SiSi = compute_ffnew(r, 2.5, params_SiSi, QF_SI, QF_SI)
        ffnew_OO = compute_ffnew(r, 2.3, params_OO, QF_O, QF_O)
    elif mode == 'smooth':
        gamma = 0.2; SR_cut = 7.98001; qf_Si = 0.33; qf_O = -qf_Si/2
        cut_SiSi = 2.4; cut_SiO = 1.3; cut_OO = 2.2
        ffnew_SiO = compute_smoothed_ff(r, R, f_SiO3, qf_Si, qf_O, cut_SiO, SR_cut, gamma)
        ffnew_SiSi = compute_smoothed_ff(r, R, f_SiSi3, qf_Si, qf_Si, 2.5, SR_cut, gamma)
        ffnew_OO = compute_smoothed_ff(r, R, f_OO3, qf_O, qf_O, cut_OO, SR_cut, gamma)
    else:
        print('error')
    save_for_lammps('lammps_ff_SiO_{}.txt'.format(mode), np.column_stack([np.arange(1, len(r)+1), ffnew_SiO]))
    save_for_lammps('lammps_ff_SiSi_{}.txt'.format(mode), np.column_stack([np.arange(1, len(r)+1), ffnew_SiSi]))
    save_for_lammps('lammps_ff_OO_{}.txt'.format(mode), np.column_stack([np.arange(1, len(r)+1), ffnew_OO]))

    # Plotting
    plot_forces(ffnew_SiSi, ffnew_SiO, ffnew_OO, R, f_SiSi3, f_SiO3, f_OO3)

if __name__ == "__main__":
    main(mode='smooth')
