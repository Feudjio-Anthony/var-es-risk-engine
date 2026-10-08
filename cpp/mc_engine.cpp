// Monte Carlo engine: generation of correlated Gaussian scenarios and
// computation of the portfolio P&L, in ONE single pass.
//
// Exposed to Python with pybind11 as the module "mc_engine".
//
// What it computes, for each scenario s = 1..n_sims:
//     z      ~ N(0, I)                 (n independent standard normals)
//     x      = mu + L z                (correlated asset returns, L L' = Sigma)
//     P&L_s  = v0 * sum_i( w_i * x_i )
//
// Two things make it fast (see the profiling notes in docs/methodology.md):
//   1. The full matrix of simulated returns is NEVER stored: each scenario
//      is reduced to one number on the fly, and the weights are folded into
//      the Cholesky factor once (one dot product per scenario).
//   2. Normal numbers come from a ziggurat sampler driven by xoshiro256++.
//      Profiling showed that sampling dominates the cost, and that the
//      standard library's std::normal_distribution (about 16 ns per number)
//      is slower than NumPy's own ziggurat (about 12 ns per number).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Arrays are converted to contiguous float64 if needed (forcecast), so the
// raw pointers below always point to a plain C-ordered block of doubles.
using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

// ----------------------------------------------------------------------
// Random numbers
// ----------------------------------------------------------------------

// splitmix64: turns ONE 64-bit seed into a stream of well-mixed 64-bit
// numbers. We only use it to fill the 256-bit state of xoshiro256++.
static inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline std::uint64_t rotl(std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro256++ (Blackman & Vigna): a fast, high-quality 64-bit generator.
struct Xoshiro256pp {
    std::uint64_t s[4];

    explicit Xoshiro256pp(std::uint64_t seed) {
        for (int i = 0; i < 4; ++i) s[i] = splitmix64(seed);
    }

    inline std::uint64_t next() {
        const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Uniform number in (0, 1): never exactly 0, so log() is always safe.
    inline double uniform() {
        return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
};

// Ziggurat sampler for the standard normal law (Marsaglia & Tsang, 2000,
// 128 layers). The area under the bell curve is cut into 128 horizontal
// layers of equal area. About 98% of the time, a random point falls in the
// "easy" part of a layer and is accepted with one comparison and one
// multiplication. The rare remaining cases (wedge and tail) use exact tests.
struct Ziggurat {
    std::uint32_t kn[128];
    double wn[128];
    double fn[128];

    static constexpr double R = 3.442619855899;   // start of the tail

    Ziggurat() {
        const double m1 = 2147483648.0;           // 2^31
        double dn = R, tn = dn;
        const double vn = 9.91256303526217e-3;    // area of each layer

        const double q = vn / std::exp(-0.5 * dn * dn);
        kn[0] = static_cast<std::uint32_t>((dn / q) * m1);
        kn[1] = 0;
        wn[0] = q / m1;
        wn[127] = dn / m1;
        fn[0] = 1.0;
        fn[127] = std::exp(-0.5 * dn * dn);

        for (int i = 126; i >= 1; --i) {
            dn = std::sqrt(-2.0 * std::log(vn / dn + std::exp(-0.5 * dn * dn)));
            kn[i + 1] = static_cast<std::uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = std::exp(-0.5 * dn * dn);
            wn[i] = dn / m1;
        }
    }

    // One standard normal number.
    inline double sample(Xoshiro256pp& rng) const {
        const std::uint64_t r = rng.next();
        // High 32 bits give a signed integer, low 7 bits give the layer.
        std::int32_t hz = static_cast<std::int32_t>(r >> 32);
        int iz = static_cast<int>(r & 127);
        // |hz| computed in 64 bits (|INT_MIN| does not fit in 32 bits).
        const std::uint32_t ahz = hz < 0 ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(hz))
                                         : static_cast<std::uint32_t>(hz);
        if (ahz < kn[iz]) return hz * wn[iz];       // fast path (~98%)
        return slow_path(rng, hz, iz);
    }

private:
    double slow_path(Xoshiro256pp& rng, std::int32_t hz, int iz) const {
        for (;;) {
            const double x = hz * wn[iz];
            if (iz == 0) {
                // Tail beyond R: exact method by Marsaglia.
                double xt, yt;
                do {
                    xt = -std::log(rng.uniform()) * (1.0 / R);
                    yt = -std::log(rng.uniform());
                } while (yt + yt < xt * xt);
                return hz > 0 ? R + xt : -R - xt;
            }
            // Wedge between two layers: accept with the exact density.
            if (fn[iz] + rng.uniform() * (fn[iz - 1] - fn[iz]) < std::exp(-0.5 * x * x))
                return x;
            // Rejected: draw a new candidate.
            const std::uint64_t r = rng.next();
            hz = static_cast<std::int32_t>(r >> 32);
            iz = static_cast<int>(r & 127);
            const std::uint32_t ahz = hz < 0 ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(hz))
                                             : static_cast<std::uint32_t>(hz);
            if (ahz < kn[iz]) return hz * wn[iz];
        }
    }
};

// ----------------------------------------------------------------------
// Linear algebra
// ----------------------------------------------------------------------

// Cholesky decomposition: sigma = L * L^T with L lower triangular.
// Cholesky-Banachiewicz algorithm, row by row. Both matrices are stored
// as flat arrays of size n*n in row-major order: element (i, j) is at i*n+j.
static std::vector<double> cholesky(const double* sigma, int n) {
    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int k = 0; k < j; ++k)
                s += L[i * n + k] * L[j * n + k];
            if (i == j) {
                const double d = sigma[i * n + i] - s;
                // A non-positive value means sigma is not positive definite
                // (for example two assets perfectly correlated). Stop with a
                // clear message instead of silently producing NaN.
                if (!(d > 0.0))
                    throw std::runtime_error(
                        "cholesky: covariance matrix is not positive definite "
                        "(pivot " + std::to_string(i) + " is " +
                        std::to_string(d) + ")");
                L[i * n + j] = std::sqrt(d);
            } else {
                L[i * n + j] = (sigma[i * n + j] - s) / L[j * n + j];
            }
        }
    }
    return L;
}

// ----------------------------------------------------------------------
// Public function
// ----------------------------------------------------------------------

// Simulates n_sims scenarios and returns directly the vector of portfolio P&L.
py::array_t<double> simulate_pnl(DoubleArray mu, DoubleArray sigma,
                                 DoubleArray weights, int n_sims,
                                 double v0, std::uint64_t seed) {
    // ---- 1. Check the shapes (a wrong shape would read random memory) ----
    if (mu.ndim() != 1 || weights.ndim() != 1)
        throw std::invalid_argument("mu and weights must be 1-D arrays");
    const int n = static_cast<int>(mu.shape(0));
    if (n < 1)
        throw std::invalid_argument("at least one asset is required");
    if (weights.shape(0) != n)
        throw std::invalid_argument("weights must have the same length as mu");
    if (sigma.ndim() != 2 || sigma.shape(0) != n || sigma.shape(1) != n)
        throw std::invalid_argument("sigma must be an n x n matrix");
    if (n_sims < 1)
        throw std::invalid_argument("n_sims must be positive");

    const double* mu_p = mu.data();
    const double* sg_p = sigma.data();
    const double* w_p = weights.data();

    const std::vector<double> L = cholesky(sg_p, n);

    // ---- 2. Pre-compute what does not depend on the scenario ----
    // The portfolio return is  w'x = w'mu + w'L z.  So we fold the weights
    // into L once: c = L' w  (c_k = sum_i w_i * L[i][k]); each scenario then
    // needs only the dot product c . z, i.e. n multiplications instead of
    // n(n+1)/2 + n.
    double port_mean = 0.0;
    for (int i = 0; i < n; ++i) port_mean += w_p[i] * mu_p[i];

    std::vector<double> c(n, 0.0);
    for (int k = 0; k < n; ++k)
        for (int i = k; i < n; ++i)          // L is lower triangular: i >= k
            c[k] += w_p[i] * L[i * n + k];

    // ---- 3. Simulate ----
    // The tables of the ziggurat are built once (thread-safe in C++11).
    static const Ziggurat zig;
    Xoshiro256pp rng(seed);

    py::array_t<double> out(n_sims);
    double* out_p = out.mutable_data();

    for (int s = 0; s < n_sims; ++s) {
        double dev = 0.0;                    // w' L z for this scenario
        for (int k = 0; k < n; ++k)
            dev += c[k] * zig.sample(rng);
        out_p[s] = v0 * (port_mean + dev);
    }
    return out;
}

PYBIND11_MODULE(mc_engine, m) {
    m.doc() = "C++ Monte Carlo engine for VaR and ES";
    m.def("simulate_pnl", &simulate_pnl,
          "Simulate the P&L of a portfolio of correlated Gaussian assets.\n\n"
          "mu: mean returns (n,), sigma: covariance (n, n), weights: (n,).\n"
          "Returns an array of n_sims P&L values (signed, in currency units).",
          py::arg("mu"), py::arg("sigma"), py::arg("weights"),
          py::arg("n_sims"), py::arg("v0") = 1e6, py::arg("seed") = 42);
}