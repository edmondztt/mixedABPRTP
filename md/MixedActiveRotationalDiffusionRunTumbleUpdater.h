// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MixedActiveRotationalDiffusionRunTumbleUpdater.h
    \brief Declares an updater that actively diffuses particle orientations
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "MixedActiveForceCompute.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"

#include <memory>
#include <vector>
#include <pybind11/pybind11.h>

#pragma once

namespace hoomd
    {
namespace md
    {
/// Updates particle's orientations based on a given diffusion constant.
/** The updater accepts a variant rotational diffusion and updates the particle orientations of the
 * associated MixedActiveForceCompute's group (by calling m_active_force.rotationalDiffusion).
 *
 * Note: This was originally part of the MixedActiveForceCompute, and is separated to obey the idea that
 * force computes do not update the system directly, but updaters do. See GitHub issue (898). The
 * updater is just a shell that calls through to m_active_force due to the complexities of the logic
 * with the introduction of manifolds.
 *
 * If anyone has the time to do so, the implementation would be cleaner if moved to this updater.
 */
class PYBIND11_EXPORT MixedActiveRotationalDiffusionRunTumbleUpdater : public Updater
    {
    public:
    /// Constructor
    MixedActiveRotationalDiffusionRunTumbleUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                     std::shared_ptr<PeriodicTrigger> trigger,
                                     std::shared_ptr<Variant> rotational_diffusion,
                                    //  std::vector<Scalar> tumble_rate,
                                     std::shared_ptr<Variant> tumble_angle_gauss_spread,
                                     std::shared_ptr<MixedActiveForceCompute> mixed_active_force,
                                     bool taxis);

    /// Destructor
    virtual ~MixedActiveRotationalDiffusionRunTumbleUpdater();

    /// Get the rotational diffusion
    std::shared_ptr<Variant>& getRotationalDiffusion()
        {
        return m_rotational_diffusion;
        }

    // Get the spread of tumble angle in the gaussian around pi
    std::shared_ptr<Variant>& getTumbleAngleGaussSpread()
        {
        return m_tumble_angle_gauss_spread;
        }

    bool& getTaxis()
        {
        return m_taxis;
        }

    /// set rotational_diffusion
    void setRotationalDiffusion(std::shared_ptr<Variant>& new_diffusion)
        {
        m_rotational_diffusion = new_diffusion;
        }
    
    /// set spread of tumble angle in the gaussian around pi 
    void setTumbleAngleGaussSpread(std::shared_ptr<Variant>& new_tumble_angle_gauss_spread)
        {
        m_tumble_angle_gauss_spread = new_tumble_angle_gauss_spread;
        }

    void setTaxis(bool iftaxis)
        {
        m_taxis = iftaxis;
        }

    std::shared_ptr<PeriodicTrigger> getPeriodicTrigger() const {
        return std::static_pointer_cast<PeriodicTrigger>(m_trigger);
    };

    // std::vector<std::shared_ptr<Variant>>& getTumbleRate()
    // {
    //     return m_tumble_rate;
    // }
    // void setTumbleRate(std::vector<std::shared_ptr<Variant>>& new_tumble_rate){
    //     m_tumble_rate = new_tumble_rate;
    // }

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    private:
    bool m_taxis;
    std::shared_ptr<Variant>
        m_rotational_diffusion; //!< Variant that determines the current rotational diffusion
    std::shared_ptr<Variant>
        m_tumble_angle_gauss_spread; //!< Variant that determines the spread of tumble angle in the gaussian around pi
    // if tumble angle gauss spread < 0, means no kinesis only random walk
    std::shared_ptr<MixedActiveForceCompute>
        m_active_force; //!< Active force to call rotationalDiffusion and tumble on
    };

    } // end namespace md
    } // end namespace hoomd
