#ifndef SOLVER_H
#define SOLVER_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 
#include <iostream>

using namespace std;


/*
class TemplateBaseClass_t
{
  public:
    CUDA_CALLABLE_MEMBER 
      virtual void operator=(TemplateBaseClass_t &_TemplateBaseClass);
    CUDA_CALLABLE_MEMBER 
      virtual TemplateBaseClass_t operator+(const TemplateBaseClass_t &_TemplateBaseClass) const ;
    CUDA_CALLABLE_MEMBER 
      virtual TemplateBaseClass_t operator-(const TemplateBaseClass_t &_TemplateBaseClass) const ;
    CUDA_CALLABLE_MEMBER 
      virtual TemplateBaseClass_t operator*(const float_tt &d) const ;
    CUDA_CALLABLE_MEMBER 
      virtual TemplateBaseClass_t operator/(const float_tt &d) const ;
    CUDA_CALLABLE_MEMBER 
      virtual TemplateBaseClass_t norm() const ;
};

template <typename BaseClass_t>
class TemplateCallableClass_t
{
  public:
    CUDA_CALLABLE_MEMBER 
      virtual BaseClass_t operator()(const BaseClass_t& _BaseClass) const;
};
*/

template<typename float_tt>
struct err_n_dt_t
{
  CUDA_CALLABLE_MEMBER
    void operator=(const err_n_dt_t<float_tt>& _end) { err = _end.err; nbs = _end.nbs; dt = _end.dt; } 
  CUDA_CALLABLE_MEMBER
    void set(float_tt _err, unsigned int _nbs, float_tt _dt){ err = _err; nbs = _nbs; dt = _dt; }
  float_tt err, dt;
  unsigned int nbs;
};

struct RKF_t
{
  RKF_t(){}
  double a21 = 1./4,
         a31 = 3./32, a32 = 9./32,
         a41 = 1932./2197, a42 = -7200./2197, a43 = 7296./2197,
         a51 = 439./216, a52 = -8., a53 = 3680./513, a54 = -845./4104,
         a61 = -8./27, a62 = 2., a63 = -3544./2565, a64 = 1859./4104, a65 = -11./40,
         b1 = 16./135, b2 = 0., b3 = 6656./12825, b4 = 28561./56430, b5 = -9./50, b6 = 2./55,
         bb1 = 25./216, bb2 = 0., bb3 = 1408./2565, bb4 = 2197./4104, bb5 = -1./5, bb6 = 0.; 
};  

template <typename CallableClass_t, typename BaseClass_t, typename float_tt>
class RK45Solver_t
{
  private :
    float_tt dt_min;
    unsigned int nb_steps_max;
    const RKF_t RKF;
    float_tt dt, err, tol;
    err_n_dt_t<float_tt> err_n_dt;
    BaseClass_t BaseClass_out_deg;
    CallableClass_t Callable;

    CUDA_CALLABLE_MEMBER
      float_tt one_step(const BaseClass_t& _BaseClass_in, BaseClass_t& _BaseClass_out);

  public:
    CUDA_CALLABLE_MEMBER
      RK45Solver_t(CallableClass_t& _Callable, float_tt _dt_min, unsigned int _nb_steps_max);
    CUDA_CALLABLE_MEMBER
      CallableClass_t& get_CallableClass() const;
    CUDA_CALLABLE_MEMBER 
      bool operator()(float_tt _dt, float_tt _tol, const BaseClass_t& _BaseClass_in, BaseClass_t& _BaseClass_out);
    CUDA_CALLABLE_MEMBER
      const err_n_dt_t<float_tt>& get_err_n_dt() const;
};


template <typename CallableClass_t, typename BaseClass_t, typename float_tt>
CUDA_CALLABLE_MEMBER
RK45Solver_t <CallableClass_t, BaseClass_t, float_tt>::RK45Solver_t(CallableClass_t& _Callable, float_tt _dt_min, unsigned int _nb_steps_max) : 
  Callable(_Callable), dt_min (_dt_min), nb_steps_max (_nb_steps_max) {}
  
template <typename CallableClass_t, typename BaseClass_t, typename float_tt>
CUDA_CALLABLE_MEMBER
float_tt RK45Solver_t<CallableClass_t, BaseClass_t, float_tt>::one_step(const BaseClass_t& _BaseClass_in, BaseClass_t& _BaseClass_out)
{
  const BaseClass_t &X = _BaseClass_in;
  BaseClass_t &Y = _BaseClass_out,
              &Ydeg = BaseClass_out_deg,
              K1, K2, K3, K4, K5, K6,
              K1int, K2int, K3int, K4int, K5int, K6int,
              dX;

  K1 = Callable(X);

  K1int = K1*RKF.a21*dt;
  K2 = Callable(K1int+X);

  K1int = K1*RKF.a31*dt;
  K2int = K2*RKF.a32*dt;
  K3 = Callable(K2int+K1int+X);
  
  K1int = K1*RKF.a41*dt;
  K2int = K2*RKF.a42*dt;
  K3int = K3*RKF.a43*dt;
  K4 = Callable(K3int+K2int+K1int+X);

  K1int = K1*RKF.a51*dt;
  K2int = K2*RKF.a52*dt;
  K3int = K3*RKF.a53*dt;
  K4int = K4*RKF.a54*dt;
  K5 = Callable(K4int+K3int+K2int+K1int+X);

  K1int = K1*RKF.a61*dt;
  K2int = K2*RKF.a62*dt;
  K3int = K3*RKF.a63*dt;
  K4int = K4*RKF.a64*dt;
  K5int = K5*RKF.a65*dt;
  K6 = Callable(K4int+K3int+K2int+K1int+X);
  
  dX = (K1*RKF.b1 + K2*RKF.b2 + K3*RKF.b3 + K4*RKF.b4 + K5*RKF.b5 + K6*RKF.b6)*dt;
  Y = X + dX;

  dX = (K1*(RKF.b1-RKF.bb1) + K2*(RKF.b2-RKF.bb2) + K3*(RKF.b3-RKF.bb3) + K4*(RKF.b4-RKF.bb4) + K5*(RKF.b5-RKF.bb5) + K6*(RKF.b6-RKF.bb6))*dt;
  Ydeg = X + dX;

  return (Y-Ydeg).norm();
}

template <typename CallableClass_t, typename BaseClass_t, typename float_tt>
CUDA_CALLABLE_MEMBER
bool RK45Solver_t<CallableClass_t, BaseClass_t, float_tt>::operator()(float_tt _dt, float_tt _tol, const BaseClass_t& _BaseClass_in, BaseClass_t& _BaseClass_out)
{ 
  dt = _dt; tol = _tol;
  int nb_steps=0;
  for(float_tt t=0; t<_dt; t+=dt)
  {
    err = one_step(_BaseClass_in, _BaseClass_out);
    while (err>tol) 
    {
      dt = .9*powf(tol/err, 1./3);
      nb_steps++;
      if (dt<dt_min || nb_steps>nb_steps_max)
      {
        err_n_dt.set(err, nb_steps, dt);
        return false;
      }
      err = one_step(_BaseClass_in, _BaseClass_out);
    } 
  }
  err_n_dt.set(err, nb_steps, dt);
  return true;
}

template <typename CallableClass_t, typename BaseClass_t, typename float_tt>
CUDA_CALLABLE_MEMBER
const err_n_dt_t<float_tt>& RK45Solver_t<CallableClass_t, BaseClass_t, float_tt>::get_err_n_dt() const { return err_n_dt; }

#endif
