#include <Eigen/Dense>

template<typename T>
using Vector = Eigen::Matrix<T, 2, 1>;

template<typename T>
using Matrix = Eigen::Matrix<T, 2, 2>;

using real = float;

inline auto diag(const float value) -> Matrix<real> {
    Matrix<real> m = Matrix<real>::Zero();
    m(0, 0) = value;
    m(1, 1) = value;
    return m;
}

inline auto constmat(const float value) -> Matrix<real> { return Matrix<real>::Constant(value); }
inline auto constvec(const float value) -> Vector<real> { return Vector<real>::Constant(value); }

inline auto nc_rand_int() -> uint32_t {
    static unsigned int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned int t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

inline auto nc_rand() -> float { return nc_rand_int() * (1.0f / 4294967296.0f); }
inline auto randvec() -> Vector<real> {
    Vector<real> ret;
    for (int i = 0; i < 2; ++i) { ret(i) = nc_rand(); }

    return ret;
}

inline auto nclr_svd(const Matrix<real> &a, Matrix<real> &U, Matrix<real> &sig, Matrix<real> &V) -> void {
    const auto svd = Eigen::JacobiSVD<Matrix<real>>(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();
    const auto values = svd.singularValues();
    for (int ii = 0; ii < values.rows(); ++ii) { sig(ii, ii) = values(ii); }
}

inline auto nclr_polar(const Matrix<real> &m, Matrix<real> &R, Matrix<real> &S) -> void {
    Matrix<real> sig;
    Matrix<real> U, V;
    nclr_svd(m, U, sig, V);

    R = U * V.transpose();
    S = V * sig * V.transpose();
}
