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
    void setActiveForce(const std::string& type_name, pybind11::tuple v);

    /// Gets active force vector for a given particle type
    pybind11::tuple getActiveForce(const std::string& type_name);

    /** Sets active torque vector for a given particle type
        @param typ Particle type to set active torque vector
        @param v The active torque vector value to set (a 3-tuple)
    */
    void setActiveTorque(const std::string& type_name, pybind11::tuple v);

    /// Gets active torque vector for a given particle type
    pybind11::tuple getActiveTorque(const std::string& type_name);

    std::shared_ptr<ParticleGroup>& getGroup()
        {
        return m_group;
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);

    //! update the speed and tumble rate
    virtual void update_dynamical_parameters();
    //! tumble
    virtual void tumble(Scalar tumble_angle_gauss_spread, uint64_t period, uint64_t timestep);
    //! whether should tumble now
    bool MixedActiveForceCompute::should_tumble(Scalar tumble_rate, Scalar time_elapse, hoomd::RandomGenerator rng);


    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles on which this force is applied
    GlobalVector<Scalar4>
        m_f_activeVec; //! active force unit vectors and magnitudes for each particle type

    GlobalVector<Scalar4>
        m_t_activeVec; //! active torque unit vectors and magnitudes for each particle type

    GlobalVector<Scalar>
        m_tumble_rate; //! tumble rate for each particle

    private:
    // Allow MixedActiveRotationalDiffusionRunTumbleUpdater to access internal methods and members of
    // MixedActiveForceCompute classes/subclasses. This is necessary to allow
    // MixedActiveRotationalDiffusionRunTumbleUpdater to call rotationalDiffusion.
    friend class MixedActiveRotationalDiffusionRunTumbleUpdater;
    };

    } // end namespace md
    } // end namespace hoomd
#endif
