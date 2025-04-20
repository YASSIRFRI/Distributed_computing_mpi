/****************************************************************************
 * Rayleigh-Taylor Instability Simulation (CPU Version)
 *
 * This program simulates the Rayleigh-Taylor instability for a compressible
 * fluid in 2D using finite differences on a grid with ghost cells.
 *
 * - The physical domain is defined with indices 1..Nx-2 in x and 1..Ny-2 in y.
 * - Ghost cells at i=0 and i=Nx-1 are used for periodic conditions in x.
 * - Ghost cells at j=0 and j=Ny-1 are used for reflective (no-normal-flow)
 *   conditions in y (v is flipped to zero at the boundaries).
 *
 * The governing equations are:
 *
 *   Continuity:
 *       ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = k1 (∂²ρ/∂x² + ∂²ρ/∂y²)
 *
 *   Momentum x:
 *       ∂(ρu)/∂t + ∂(ρu² + p)/∂x + ∂(ρuv)/∂y = k2 (∂²(ρu)/∂x² + ∂²(ρu)/∂y²)
 *
 *   Momentum y:
 *       ∂(ρv)/∂t + ∂(ρuv)/∂x + ∂(ρv²+p)/∂y = -GA*ρ + k2 (∂²(ρv)/∂x² + ∂²(ρv)/∂y²)
 *
 *   Energy:
 *       ∂e/∂t + ∂[u(e+p)]/∂x + ∂[v(e+p)]/∂y = -GA*ρ*v + k3 (∂²e/∂x² + ∂²e/∂y²)
 *
 * where the pressure is computed from the conservative variables:
 *       p = (GAM - 1)[e - 0.5 ρ (u²+v²)]
 *
 * INITIAL CONDITIONS (in the physical domain):
 *   - The interface is at y = Ly/2.
 *   - ρ = 2 for y ≥ Ly/2; ρ = 1 for y < Ly/2.
 *   - u = 0 everywhere.
 *   - v is zero except near the interface (|y - Ly/2| ≤ 0.05) where v is 
 *     set to a random value in ±1e-3.
 *   - p = 40 + ρ * GA * (y - Ly/2)
 *   - e is computed from p.
 *
 * The time step is chosen to satisfy a CFL condition:
 *       dt = CFL * min(dx,dy) / max(|u|, |v|, c)
 * where c = sqrt(GAM * p/ρ).
 *
 * The simulation uses a two-stage Runge-Kutta (RK2) integrator.
 *
 * The final state (only physical cells) is output in ASCII to:
 *    density_final.dat, u_final.dat, v_final.dat, energy_final.dat
 *
 * Global constants used in this program:
 *    GA   = -10.0       (gravitational acceleration)
 *    GAM  = 1.4         (ratio of specific heats)
 *    CFL  = 0.2         (CFL number)
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <algorithm>

// Define missing constants.
const double GA  = -10.0;
const double GAM = 1.4;
const double CFL = 0.2;

// -----------------------------
// Grid structure:
// The physical domain has phys_Nx x phys_Ny cells.
// We add one ghost cell on each side, so total Nx = phys_Nx+2, Ny = phys_Ny+2.
// -----------------------------
struct grid {
    double Lx, Ly;      // Physical domain dimensions.
    int phys_Nx, phys_Ny; // Number of physical cells.
    int Nx, Ny;         // Total cells including ghost layers.
    double dx, dy;      // Grid spacing (computed with physical size).
    int N;              // Total number of cells = Nx * Ny.
};

// -----------------------------
// Initialize grid.
// -----------------------------
void init_grid(grid &g, int phys_Nx, int phys_Ny, double Lx, double Ly) {
    g.Lx = Lx;
    g.Ly = Ly;
    g.phys_Nx = phys_Nx;
    g.phys_Ny = phys_Ny;
    g.Nx = phys_Nx + 2; // ghost cells left and right.
    g.Ny = phys_Ny + 2; // ghost cells bottom and top.
    g.dx = Lx / double(phys_Nx);
    g.dy = Ly / double(phys_Ny);
    g.N  = g.Nx * g.Ny;
}

inline double compute_pressure(double r, double ru, double rv, double e) {
    double u = ru / r;
    double v = rv / r;
    double kinetic = 0.5 * r * (u*u + v*v);
    double p = (GAM - 1.0) * (e - kinetic);
    return (p < 1.0e-14 ? 1.0e-14 : p);
}

// -----------------------------
// Apply boundary conditions.
// - Periodic in x: ghost cells at i=0 equal to cell at i=Nx-2; at i=Nx-1 equal to cell at i=1.
// - Reflective in y: ghost cells at j=0 and j=Ny-1 copy interior values and flip sign on rv.
// -----------------------------
void apply_BCs(const grid &g, double *r, double *ru, double *rv, double *E) {
    // Periodic in x for all rows.
    for (int j = 0; j < g.Ny; j++) {
        int row = j * g.Nx;
        r[row]       = r[row + g.Nx - 2];  // left ghost <- last physical cell.
        ru[row]      = ru[row + g.Nx - 2];
        rv[row]      = rv[row + g.Nx - 2];
        E[row]       = E[row + g.Nx - 2];
        r[row + g.Nx - 1] = r[row + 1];      // right ghost <- first physical cell.
        ru[row + g.Nx - 1] = ru[row + 1];
        rv[row + g.Nx - 1] = rv[row + 1];
        E[row + g.Nx - 1] = E[row + 1];
    }
    // Reflective in y for all columns.
    for (int i = 0; i < g.Nx; i++) {
        // Bottom ghost row (j = 0) from j = 1.
        r[i] = r[g.Nx + i];
        ru[i] = ru[g.Nx + i];
        rv[i] = -rv[g.Nx + i]; // flip vertical momentum.
        E[i] = E[g.Nx + i];
        // Top ghost row (j = Ny - 1) from j = Ny - 2.
        r[(g.Ny-1)*g.Nx + i] = r[(g.Ny-2)*g.Nx + i];
        ru[(g.Ny-1)*g.Nx + i] = ru[(g.Ny-2)*g.Nx + i];
        rv[(g.Ny-1)*g.Nx + i] = -rv[(g.Ny-2)*g.Nx + i]; // flip sign.
        E[(g.Ny-1)*g.Nx + i] = E[(g.Ny-2)*g.Nx + i];
    }
}

// -----------------------------
// Initial conditions.
// Physical domain indices: i = 1 .. Nx-2, j = 1 .. Ny-2.
// The interface is at y = Ly/2. For y >= Ly/2: ρ=2, otherwise ρ=1.
// u = 0 everywhere.
// v is assigned a small random perturbation (±1e-3) if |y - Ly/2| <= 0.05.
// p = 40 + ρ * GA * (y - Ly/2).
// E = p/(GAM-1) + 0.5ρ(u²+v²).
// -----------------------------
void initial_conditions(const grid &g, double *r, double *ru, double *rv, double *E) {
    srand((unsigned)time(NULL));
    double ymid = 0.5 * g.Ly;
    // Loop over physical cells
    for (int j = 1; j < g.Ny - 1; j++) {
        // Map ghost-adjusted j to physical y coordinate:
        double y = (j - 1 + 0.5) * g.dy; 
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            double dens = (y >= ymid) ? 2.0 : 1.0;
            r[idx] = dens;
            double u = 0.0;
            double v = 0.0;
            if (fabs(y - ymid) <= 0.05)
                v = ((double)rand() / RAND_MAX - 0.5) * 2.0e-3; // ±1e-3
            ru[idx] = dens * u;
            rv[idx] = dens * v;
            double p0 = 40.0 + dens * GA * (y - ymid);
            if (p0 < 1e-14) p0 = 1e-14;
            double ke = 0.5 * dens * (u*u + v*v);
            E[idx] = p0 / (GAM - 1.0) + ke;
        }
    }
}

// -----------------------------
// Compute dt from the CFL condition.
// dt = CFL * (min(dx,dy)) / max(|u|, |v|, c), where c = sqrt(GAM * p / ρ)
// Only consider the physical domain.
double compute_dt(const grid &g, const double *r, const double *ru, const double *rv, const double *E) {
    double max_speed = 1e-14;
    for (int j = 1; j < g.Ny - 1; j++) {
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            double rr = r[idx];
            if (rr < 1e-14) continue;
            double u = ru[idx] / rr;
            double v = rv[idx] / rr;
            double p = compute_pressure(rr, ru[idx], rv[idx], E[idx]);
            double c = sqrt(GAM * p / rr);
            double local = std::max({fabs(u), fabs(v), c});
            if (local > max_speed)
                max_speed = local;
        }
    }
    double h = std::min(g.dx, g.dy);
    double dt = CFL * (h / max_speed);
    if (max_speed < 1e-10)
        printf("WARNING: max_speed = %.4g is extremely small => dt = %.4g\n", max_speed, dt);
    return dt;
}

// -----------------------------
// Compute RHS of the PDEs (Euler form) using central differences.
// Computations are done in the physical domain (i = 1..Nx-2, j = 1..Ny-2).
//
// Diffusion terms use coefficients k1, k2, k3.
// -----------------------------
void compute_RHS(const grid &g, const double *r, const double *ru, const double *rv,
                 const double *E, double *rhs_r, double *rhs_ru,
                 double *rhs_rv, double *rhs_E,
                 double k1, double k2, double k3)
{
    int Nx = g.Nx, Ny = g.Ny;
    double dx = g.dx, dy = g.dy;
    double dx2 = dx * dx, dy2 = dy * dy;
    for (int j = 1; j < Ny - 1; j++) {
        for (int i = 1; i < Nx - 1; i++) {
            int idx = j * Nx + i;
            // Continuity: ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = k1 Δρ
            double d_ru_dx = (ru[idx+1] - ru[idx-1]) / (2*dx);
            double d_rv_dy = (rv[idx+Nx] - rv[idx-Nx]) / (2*dy);
            double lap_r = (r[idx+1] - 2*r[idx] + r[idx-1]) / dx2 +
                           (r[idx+Nx] - 2*r[idx] + r[idx-Nx]) / dy2;
            rhs_r[idx] = - (d_ru_dx + d_rv_dy) + k1 * lap_r;

            // Momentum x:
            // u = ru/ρ; flux in x: ρu² + p, flux in y: ρuv.
            double u = ru[idx] / r[idx];
            double p_local = compute_pressure(r[idx], ru[idx], rv[idx], E[idx]);
            double F_im1 = (ru[idx-1] / r[idx-1]) * ru[idx-1] + compute_pressure(r[idx-1], ru[idx-1], rv[idx-1], E[idx-1]);
            double F_ip1 = (ru[idx+1] / r[idx+1]) * ru[idx+1] + compute_pressure(r[idx+1], ru[idx+1], rv[idx+1], E[idx+1]);
            double dF_dx = (F_ip1 - F_im1) / (2*dx);
            double G_jm1 = (ru[idx-Nx] / r[idx-Nx]) * rv[idx-Nx];
            double G_jp1 = (ru[idx+Nx] / r[idx+Nx]) * rv[idx+Nx];
            double dG_dy = (G_jp1 - G_jm1) / (2*dy);
            double lap_ru = (ru[idx+1] - 2*ru[idx] + ru[idx-1]) / dx2 +
                            (ru[idx+Nx] - 2*ru[idx] + ru[idx-Nx]) / dy2;
            rhs_ru[idx] = - (dF_dx + dG_dy) + k2 * lap_ru;

            // Momentum y:
            // v = rv/ρ; flux in x: ρuv, flux in y: ρv²+p.
            double v = rv[idx] / r[idx];
            double H_im1 = (ru[idx-1] / r[idx-1]) * rv[idx-1];
            double H_ip1 = (ru[idx+1] / r[idx+1]) * rv[idx+1];
            double dH_dx = (H_ip1 - H_im1) / (2*dx);
            double I_jm1 = (rv[idx-Nx] / r[idx-Nx]) * rv[idx-Nx] + compute_pressure(r[idx-Nx], ru[idx-Nx], rv[idx-Nx], E[idx-Nx]);
            double I_jp1 = (rv[idx+Nx] / r[idx+Nx]) * rv[idx+Nx] + compute_pressure(r[idx+Nx], ru[idx+Nx], rv[idx+Nx], E[idx+Nx]);
            double dI_dy = (I_jp1 - I_jm1) / (2*dy);
            double lap_rv = (rv[idx+1] - 2*rv[idx] + rv[idx-1]) / dx2 +
                            (rv[idx+Nx] - 2*rv[idx] + rv[idx-Nx]) / dy2;
            rhs_rv[idx] = - (dH_dx + dI_dy) + r[idx] * GA + k2 * lap_rv;

            // Energy:
            // Flux in x: u(e+p), in y: v(e+p)
            double ep = E[idx] + p_local;
            double flux_ex_im1 = (ru[idx-1]/r[idx-1]) * (E[idx-1] + compute_pressure(r[idx-1], ru[idx-1], rv[idx-1], E[idx-1]));
            double flux_ex_ip1 = (ru[idx+1]/r[idx+1]) * (E[idx+1] + compute_pressure(r[idx+1], ru[idx+1], rv[idx+1], E[idx+1]));
            double d_flux_ex_dx = (flux_ex_ip1 - flux_ex_im1) / (2*dx);
            double flux_ey_jm1 = (rv[idx-Nx]/r[idx-Nx]) * (E[idx-Nx] + compute_pressure(r[idx-Nx], ru[idx-Nx], rv[idx-Nx], E[idx-Nx]));
            double flux_ey_jp1 = (rv[idx+Nx]/r[idx+Nx]) * (E[idx+Nx] + compute_pressure(r[idx+Nx], ru[idx+Nx], rv[idx+Nx], E[idx+Nx]));
            double d_flux_ey_dy = (flux_ey_jp1 - flux_ey_jm1) / (2*dy);
            double lap_E = (E[idx+1] - 2*E[idx] + E[idx-1]) / dx2 +
                           (E[idx+Nx] - 2*E[idx] + E[idx-Nx]) / dy2;
            rhs_E[idx] = - (d_flux_ex_dx + d_flux_ey_dy) - r[idx] * GA * v + k3 * lap_E;
        }
    }
}

// -----------------------------
// Advance the solution by one Euler time step.
// -----------------------------
void advance_Euler(int N, double *q, const double *rhs, double dt) {
    for (int i = 0; i < N; i++) {
        q[i] += dt * rhs[i];
    }
}

// -----------------------------
// Final RK2 combination: q_new = 0.5*(q_old + (q_star + dt * rhs(q_star)))
// -----------------------------
void final_RK2(int N, const double *q_old, const double *q_star,
               const double *rhs_star, double *q_new, double dt)
{
    for (int i = 0; i < N; i++) {
        q_new[i] = 0.5 * ( q_old[i] + (q_star[i] + dt * rhs_star[i]) );
    }
}

// -----------------------------
// Main function.
// -----------------------------
int main() {
    // Set physical grid size and domain.
    int phys_Nx = 128;
    int phys_Ny = 256;
    double Lx = 1.0, Ly = 1.0;
    grid g;
    init_grid(g, phys_Nx, phys_Ny, Lx, Ly);
    printf("Grid: Physical = %d x %d, Total (with ghosts) = %d x %d\n",
           g.phys_Nx, g.phys_Ny, g.Nx, g.Ny);
    printf("dx = %.3g, dy = %.3g\n", g.dx, g.dy);

    int N = g.N;
    // Allocate arrays.
    double *r     = new double[N];
    double *ru    = new double[N];
    double *rv    = new double[N];
    double *E     = new double[N];
    // For RK2 stage storage.
    double *r_star  = new double[N];
    double *ru_star = new double[N];
    double *rv_star = new double[N];
    double *E_star  = new double[N];
    double *r_new   = new double[N];
    double *ru_new  = new double[N];
    double *rv_new  = new double[N];
    double *E_new   = new double[N];
    // RHS arrays.
    double *rhs_r   = new double[N];
    double *rhs_ru  = new double[N];
    double *rhs_rv  = new double[N];
    double *rhs_E   = new double[N];
    double *rhs_star_r  = new double[N];
    double *rhs_star_ru = new double[N];
    double *rhs_star_rv = new double[N];
    double *rhs_star_E  = new double[N];

    // Set initial conditions.
    initial_conditions(g, r, ru, rv, E);
    apply_BCs(g, r, ru, rv, E);

    double time = 0.0;
    int num_steps = 3000;
    for (int step = 0; step < num_steps; step++) {
        // Apply boundary conditions.
        apply_BCs(g, r, ru, rv, E);

        // Compute dt.
        double dt = compute_dt(g, r, ru, rv, E);
        
        // For diffusion coefficients:
        double D = (g.dx * g.dx) / (2.0 * dt);
        double k1 = 0.0125 * D;
        double k2 = 0.1250 * D;
        double k3 = 0.0125 * D;

        // Save q^n.
        double *r_old  = new double[N];
        double *ru_old = new double[N];
        double *rv_old = new double[N];
        double *E_old  = new double[N];
        memcpy(r_old, r, N * sizeof(double));
        memcpy(ru_old, ru, N * sizeof(double));
        memcpy(rv_old, rv, N * sizeof(double));
        memcpy(E_old, E, N * sizeof(double));

        // Stage 1: Compute RHS(q^n) and update by Euler.
        std::fill(rhs_r, rhs_r + N, 0.0);
        std::fill(rhs_ru, rhs_ru + N, 0.0);
        std::fill(rhs_rv, rhs_rv + N, 0.0);
        std::fill(rhs_E, rhs_E + N, 0.0);
        compute_RHS(g, r, ru, rv, E, rhs_r, rhs_ru, rhs_rv, rhs_E, k1, k2, k3);
        advance_Euler(N, r, rhs_r, dt);
        advance_Euler(N, ru, rhs_ru, dt);
        advance_Euler(N, rv, rhs_rv, dt);
        advance_Euler(N, E, rhs_E, dt);
        // Save q_star.
        for (int i = 0; i < N; i++) {
            r_star[i] = r[i];
            ru_star[i] = ru[i];
            rv_star[i] = rv[i];
            E_star[i] = E[i];
        }
        // Revert to q^n.
        for (int i = 0; i < N; i++) {
            r_new[i] = r[i] - dt * rhs_r[i];
            ru_new[i] = ru[i] - dt * rhs_ru[i];
            rv_new[i] = rv[i] - dt * rhs_rv[i];
            E_new[i] = E[i] - dt * rhs_E[i];
        }
        memcpy(r, r_new, N * sizeof(double));
        memcpy(ru, ru_new, N * sizeof(double));
        memcpy(rv, rv_new, N * sizeof(double));
        memcpy(E, E_new, N * sizeof(double));
        // Apply BC to q_star.
        apply_BCs(g, r_star, ru_star, rv_star, E_star);

        // Stage 2: Compute RHS(q_star) and final combination.
        std::fill(rhs_star_r, rhs_star_r + N, 0.0);
        std::fill(rhs_star_ru, rhs_star_ru + N, 0.0);
        std::fill(rhs_star_rv, rhs_star_rv + N, 0.0);
        std::fill(rhs_star_E, rhs_star_E + N, 0.0);
        compute_RHS(g, r_star, ru_star, rv_star, E_star,
                    rhs_star_r, rhs_star_ru, rhs_star_rv, rhs_star_E,
                    k1, k2, k3);
        for (int i = 0; i < N; i++) {
            r_new[i] = 0.5 * ( r_old[i] + (r_star[i] + dt * rhs_star_r[i]) );
            ru_new[i] = 0.5 * ( ru_old[i] + (ru_star[i] + dt * rhs_star_ru[i]) );
            rv_new[i] = 0.5 * ( rv_old[i] + (rv_star[i] + dt * rhs_star_rv[i]) );
            E_new[i] = 0.5 * ( E_old[i] + (E_star[i] + dt * rhs_star_E[i]) );
        }
        memcpy(r, r_new, N * sizeof(double));
        memcpy(ru, ru_new, N * sizeof(double));
        memcpy(rv, rv_new, N * sizeof(double));
        memcpy(E, E_new, N * sizeof(double));

        delete[] r_old; delete[] ru_old; delete[] rv_old; delete[] E_old;

        time += dt;
        printf("Step %d, time = %.5e, dt = %.5e\n", step+1, time, dt);
    }

    // Write final physical domain (i=1..Nx-2, j=1..Ny-2) to ASCII files.
    FILE *fp = fopen("density_final.dat", "w");
    for (int j = 1; j < g.Ny - 1; j++) {
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            fprintf(fp, "%.6e ", r[idx]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    fp = fopen("u_final.dat", "w");
    for (int j = 1; j < g.Ny - 1; j++) {
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            double u_val = ru[idx] / r[idx];
            fprintf(fp, "%.6e ", u_val);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    fp = fopen("v_final.dat", "w");
    for (int j = 1; j < g.Ny - 1; j++) {
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            double v_val = rv[idx] / r[idx];
            fprintf(fp, "%.6e ", v_val);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    fp = fopen("energy_final.dat", "w");
    for (int j = 1; j < g.Ny - 1; j++) {
        for (int i = 1; i < g.Nx - 1; i++) {
            int idx = j * g.Nx + i;
            fprintf(fp, "%.6e ", E[idx]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    // Clean up.
    delete[] r; delete[] ru; delete[] rv; delete[] E;
    delete[] r_star; delete[] ru_star; delete[] rv_star; delete[] E_star;
    delete[] r_new; delete[] ru_new; delete[] rv_new; delete[] E_new;
    delete[] rhs_r; delete[] rhs_ru; delete[] rhs_rv; delete[] rhs_E;
    delete[] rhs_star_r; delete[] rhs_star_ru;
    delete[] rhs_star_rv; delete[] rhs_star_E;

    printf("Simulation complete. Check the final .dat files.\n");
    return 0;
}
