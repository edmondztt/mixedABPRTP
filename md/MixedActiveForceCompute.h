// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/VectorMath.h"
#include <memory>
#include "hoomd/RandomNumbers.h"

/*! \file MixedActiveForceCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __MIXEDACTIVEFORCECOMPUTE_H__
#define __MIXEDACTIVEFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {

struct mixedactive_params{
    Scalar kT1;
    Scalar kT2;
    Scalar kH1;
    Scalar kH2;
    Scalar kS1;
    Scalar kS2;
    Scalar Q0; // lower threshold for gamma
    Scalar Q1; // upper threshold for U
    Scalar noise_Q;
    Scalar U0;
    Scalar U1;
    Scalar gamma0;
    Scalar c0_PHD;

#ifndef __HIPCC__
    mixedactive_params() : kT1(1.0/600),kT2(1.0),kH1(0.1),kH2(1.0),kS1(1.0/30),
    kS2(0.1),Q0(0.3),Q1(0.7),noise_Q(0.02),U0(20),U1(10),gamma0(0.1),c0_PHD(0.1e-5) { }

    mixedactive_params(pybind11::dict params)
        : kT1(params["kT1"].cast<Scalar>()), kT2(params["kT2"].cast<Scalar>()),
        kH1(params["kH1"].cast<Scalar>()), kH2(params["kH2"].cast<Scalar>()),
        kS1(params["kS1"].cast<Scalar>()), kS2(params["kS2"].cast<Scalar>()),
        Q0(params["Q0"].cast<Scalar>()), Q1(params["Q1"].cast<Scalar>()),
        noise_Q(params["noise_Q"].cast<Scalar>()), 
        U0(params["U0"].cast<Scalar>()), U1(params["U1"].cast<Scalar>()),
        gamma0(params["gamma0"].cast<Scalar>()), c0_PHD(params["c0_PHD"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v['kT1'] = kT1;
        v['kT2'] = kT2;
        v['kH1'] = kH1;
        v['kH2'] = kH2;
        v['kS1'] = kS1;
        v['kS2'] = kS2;
        v['Q0)'] = Q0; 
        v['Q1)'] = Q1;
        v['noie_Q'] = noise_Q;
        v['U0)'] = U0;
        v['U1)'] = U1
        v['gamma0'] = gamma0;
        v['c0_PHD'] = c0_PHD;
        return v;
        }
#endif
}
// Forward declaration is necessary to avoid circular includes between this and
// MixedActiveRotationalDiffusionRunTumbleUpdater.h while making MixedActiveRotationalDiffusionRunTumbleUpdater a friend class
// of MixedActiveForceCompute.
class MixedActiveRotationalDiffusionRunTumbleUpdater;

//! Adds an active force to a number of particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT MixedActiveForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    MixedActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group);

    //! Destructor
    ~MixedActiveForceCompute();

    /** Sets active force vector for a given particle type
        @param typ Particle type to set active force vector
        @param v The active force vector value to set (a 3-tuple)
    */
    void setMixedActiveForce(const std::string& type_name, pybind11::tuple v);

    /// Gets active force vector for a given particle type
    pybind11::tuple getMixedActiveForce(const std::string& type_name);

    /** Sets active torque vector for a given particle type
        @param typ Particle type to set active torque vector
        @param v The active torque vector value to set (a 3-tuple)
    */
    void setActiveTorque(const std::string& type_name, pybind11::tuple v);

    /// Gets active torque vector for a given particle type
    pybind11::tuple getActiveTorque(const std::string& type_name);

    virtual void setParams(unsigned int type, Scalar m_kT1,
    Scalar m_kT2,
    Scalar m_kH1,
    Scalar m_kH2,
    Scalar m_kS1,
    Scalar m_kS2,
    Scalar m_Q0, 
    Scalar m_Q1, 
    Scalar m_noise_Q,
    Scalar m_U0,
    Scalar m_U1,
    Scalar m_gamma0,
    Scalar m_c0_PHD);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a given type
    virtual pybind11::dict getParams(std::string type);

    std::shared_ptr<ParticleGroup>& getGroup()
        {
        return m_group;
        }

    /*********** now for protected ***************/
    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);
    //! tumble
    virtual void tumble(Scalar tumble_angle_gauss_spread, uint64_t period, uint64_t timestep);
    //! whether should tumble now
    bool should_tumble(Scalar tumble_rate, Scalar time_elapse, hoomd::RandomGenerator rng);

    //! update the speed and tumble rate
    virtual void update_dynamical_parameters();

    void update_Q(Scalar &Q, Scalar c_new, Scalar c_old, int FLAG_Q, unsigned int typ);
    void update_S(Scalar &S, Scalar gamma, unsigned int typ);
    void update_U(Scalar &U, Scalar Q, unsigned int typ);
    void update_tumble_rate(Scalar &gamma, Scalar Q, unsigned int typ);
    Scalar compute_c_new(Scalar4 pos);

    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles on which this force is applied
    GlobalVector<Scalar4>
        m_f_activeVec; //! active force unit vectors and magnitudes for each particle type

    GlobalVector<Scalar4>
        m_t_activeVec; //! active torque unit vectors and magnitudes for each particle type

    // by each particle
    GlobalVector<Scalar> m_tumble_rate; //! tumble rate for each particle
    GlobalVector<Scalar> m_U;
    GlobalVector<Scalar> m_QH;
    GlobalVector<Scalar> m_QT;
    GlobalVector<Scalar> m_S;
    GlobalVector<Scalar> m_c; // c_old

    Scalar m_dt;
    // by type:
    Scalar* m_kT1;
    Scalar* m_kT2;
    Scalar* m_kH1;
    Scalar* m_kH2;
    Scalar* m_kS1;
    Scalar* m_kS2;
    Scalar* m_Q0; // lower threshold for gamma
    Scalar* m_Q1; // upper threshold for U
    Scalar* m_noise_Q;
    Scalar* m_U0;
    Scalar* m_U1;
    Scalar* m_gamma0;
    Scalar* m_c0_PHD;

    private:
    static constexpr int m_FLAG_QH = 1; // for QH
    static constexpr int m_FLAG_QT = 2; // for QT
    // Allow MixedActiveRotationalDiffusionRunTumbleUpdater to access internal methods and members of
    // MixedActiveForceCompute classes/subclasses. This is necessary to allow
    // MixedActiveRotationalDiffusionRunTumbleUpdater to call rotationalDiffusion.
    friend class MixedActiveRotationalDiffusionRunTumbleUpdater;
    };

    } // end namespace md
    } // end namespace hoomd
#endif
