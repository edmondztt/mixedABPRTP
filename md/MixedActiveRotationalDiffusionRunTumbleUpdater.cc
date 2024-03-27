// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MixedActiveRotationalDiffusionRunTumbleUpdater.cc
    \brief Defines the MixedActiveRotationalDiffusionRunTumbleUpdater class
*/

#include "MixedActiveRotationalDiffusionRunTumbleUpdater.h"

#include <iostream>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System definition
 *  \param rotational_diffusion The diffusion across time
 *  \param tumble The tumble motion
 *  \param group the particles to diffusion rotation on
 */

MixedActiveRotationalDiffusionRunTumbleUpdater::MixedActiveRotationalDiffusionRunTumbleUpdater(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<PeriodicTrigger> trigger,
    std::shared_ptr<Variant> rotational_diffusion,
    // std::vector<Scalar> tumble_rate,
    std::shared_ptr<Variant> tumble_angle_gauss_spread,
    std::shared_ptr<MixedActiveForceCompute> mixed_active_force,
    bool taxis)
    : Updater(sysdef, trigger), m_rotational_diffusion(rotational_diffusion), m_tumble_angle_gauss_spread(tumble_angle_gauss_spread), m_active_force(mixed_active_force), m_taxis(taxis)
    {
    assert(m_pdata);
    assert(m_rotational_diffusion);
    // assert(m_tumble_rate);
    assert(m_tumble_angle_gauss_spread);
    assert(m_active_force);
    printf("in MixedActiveRotationalDiffusionRunTumbleUpdater taxis=%d\n",m_taxis);
    m_exec_conf->msg->notice(5) << "Constructing MixedActiveRotationalDiffusionRunTumbleUpdater" << endl;
    }

MixedActiveRotationalDiffusionRunTumbleUpdater::~MixedActiveRotationalDiffusionRunTumbleUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying MixedActiveRotationalDiffusionRunTumbleUpdater" << endl;
    }

/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void MixedActiveRotationalDiffusionRunTumbleUpdater::update(uint64_t timestep){
    std::shared_ptr<PeriodicTrigger> trigger = getPeriodicTrigger();
    uint64_t period = trigger->getPeriod();
    m_active_force->rotationalDiffusion(m_rotational_diffusion->operator()(timestep), timestep);
    if(m_tumble_angle_gauss_spread<0){
        m_active_force->random_turn(period, timestep);
    }
    else{
        m_active_force->tumble(m_tumble_angle_gauss_spread->operator()(timestep), period, timestep);
    }
    if(m_taxis){
        m_active_force->taxisturn(timestep);
    }
}

namespace detail
    {
void export_MixedActiveRotationalDiffusionRunTumbleUpdater(pybind11::module& m)
    {
    pybind11::class_<MixedActiveRotationalDiffusionRunTumbleUpdater,
                     Updater,
                     std::shared_ptr<MixedActiveRotationalDiffusionRunTumbleUpdater>>(
        m,
        "MixedActiveRotationalDiffusionRunTumbleUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<PeriodicTrigger>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<MixedActiveForceCompute>,
                            bool>())
        .def_property("rotational_diffusion",
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::getRotationalDiffusion,
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::setRotationalDiffusion)
        .def_property("tumble_angle_gauss_spread",
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::getTumbleAngleGaussSpread,
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::setTumbleAngleGaussSpread)
        .def_property("taxis",
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::getTaxis,
                      &MixedActiveRotationalDiffusionRunTumbleUpdater::setTaxis);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
