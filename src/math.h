#include <Eigen/Dense>
using Vec = Eigen::Vector2f;
using Mat = Eigen::Matrix2f;
using nc_real = float;

inline auto diag(const float value) -> Mat {
    Mat m = Mat::Zero();
    m(0, 0) = value;
    m(1, 1) = value;
    return m;
}

inline auto constmat(const float value) -> Mat { return Mat::Constant(value); }
inline auto constvec(const float value) -> Vec { return Vec::Constant(value); }


inline auto polar_decomp(const Mat &m, Mat &R, Mat &S) -> void {
    auto x = m(0, 0) + m(1, 1);
    auto y = m(1, 0) - m(0, 1);
    auto scale = 1.0f / std::sqrt(x * x + y * y);
    auto c = x * scale, s = y * scale;
    R(0, 0) = c;
    R(0, 1) = -s;
    R(1, 0) = s;
    R(1, 1) = c;
    S = R.transpose() * m;
}

inline auto svd(const Mat &m, Mat &U, Mat &sig, Mat &V) -> void {
    Mat S;
    polar_decomp(m, U, S);
    nc_real c, s;
    if (std::abs(S(0, 1)) < 1e-6f) {
        sig = S;
        c = 1;
        s = 0;
    } else {
        auto tao = 0.5f * (S(0, 0) - S(1, 1));
        auto w = std::sqrt(tao * tao + S(0, 1) * S(0, 1));
        auto t = tao > 0 ? S(0, 1) / (tao + w) : S(0, 1) / (tao - w);
        c = 1.0f / std::sqrt(t * t + 1);
        s = -t * c;
        sig(0, 0) = std::pow(c, 2) * S(0, 0) - 2 * c * s * S(0, 1) + std::pow(s, 2) * S(1, 1);
        sig(1, 1) = std::pow(s, 2) * S(0, 0) + 2 * c * s * S(0, 1) + std::pow(c, 2) * S(1, 1);
    }
    if (sig(0, 0) < sig(1, 1)) {
        std::swap(sig(0, 0), sig(1, 1));
        V(0, 0) = -s;
        V(0, 1) = -c;
        V(1, 0) = c;
        V(1, 1) = -s;
    } else {
        V(0, 0) = c;
        V(0, 1) = -s;
        V(1, 0) = s;
        V(1, 1) = c;
    }
    V.transposeInPlace();
    U = U * V;
}


inline auto nc_rand_int() -> uint32_t {
    static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned int t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

inline auto nc_rand() -> float { return nc_rand_int() * (1.0f / 4294967296.0f); }
inline auto randvec() -> Vec {
    Vec ret;
    for (int i = 0; i < 2; ++i) { ret(i) = nc_rand(); }

    return ret;
}
