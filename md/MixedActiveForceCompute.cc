// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MixedActiveForceCompute.h"
#include "hoomd/RNGIdentifiers.h"

#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace hoomd
    {
namespace md
    {
/*! \file MixedActiveForceCompute.cc
    \brief Contains code for the MixedActiveForceCompute class
*/

/*! \param rotation_diff rotational diffusion constant for all particles.
    \param tumble_rate
 */
MixedActiveForceCompute::MixedActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Scalar L, bool is_klinokinesis, bool is_orthokinesis)
    : ForceCompute(sysdef), m_group(group), m_grid_data(std::make_unique<RectGridData>(-L/2,L/2,-L/2,L/2)), m_klinokinesis(is_klinokinesis), m_orthokinesis(is_orthokinesis) {
    
    // allocate memory for the per-type mixed_active_force storage and initialize them to (1.0,0,0)
    GlobalVector<Scalar4> tmp_f_activeVec(m_pdata->getNTypes(), m_exec_conf);
    m_f_activeVec.swap(tmp_f_activeVec);
    TAG_ALLOCATION(m_f_activeVec);

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec,
                                       access_location::host,
                                       access_mode::overwrite);
    for (unsigned int i = 0; i < m_f_activeVec.size(); i++)
        h_f_activeVec.data[i] = make_scalar4(1.0, 0.0, 0.0, 1.0);

    // allocate memory for the per-type mixed_active_force storage and initialize them to (0,0,0)
    GlobalVector<Scalar4> tmp_t_activeVec(m_pdata->getNTypes(), m_exec_conf);

    m_t_activeVec.swap(tmp_t_activeVec);
    TAG_ALLOCATION(m_t_activeVec);

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec,
                                       access_location::host,
                                       access_mode::overwrite);
    for (unsigned int i = 0; i < m_t_activeVec.size(); i++)
        h_t_activeVec.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);

    // allocate memory for the per-type tumble rate and initialize to 0.0
    unsigned int max_num_particles = m_pdata->getMaxN();
    // // tumble rate initialize
    // GlobalVector<Scalar> tmp_tumble_rate(max_num_particles, m_exec_conf);
    // m_tumble_rate.swap(tmp_tumble_rate);
    // TAG_ALLOCATION(m_tumble_rate);
    // ArrayHandle<Scalar> h_tumble_rate(m_tumble_rate,
    //                                    access_location::host,
    //                                    access_mode::overwrite);
    // for (unsigned int i = 0; i < m_tumble_rate.size(); i++)
    //     h_tumble_rate.data[i] = 0.0;
    // U0 initialize
    GlobalVector<Scalar> tmp_U(max_num_particles, m_exec_conf);
    m_U.swap(tmp_U);
    TAG_ALLOCATION(m_U);
    ArrayHandle<Scalar> h_U(m_U,
                            access_location::host,
                            access_mode::overwrite);
    for (unsigned int i = 0; i < m_U.size(); i++)
        h_U.data[i] = 0.2; // initialize velocities to all be 200 um/s
    // QH initialize
    // now these are moved to particle data
    // GlobalVector<Scalar> tmp_QH(max_num_particles, m_exec_conf);
    // m_QH.swap(tmp_QH);
    // TAG_ALLOCATION(m_QH);
    // ArrayHandle<Scalar> h_QH(m_QH,
    //                         access_location::host,
    //                         access_mode::overwrite);
    // for (unsigned int i = 0; i < m_QH.size(); i++)
    //     h_QH.data[i] = 0.0;
    // // QT initialize
    // GlobalVector<Scalar> tmp_QT(max_num_particles, m_exec_conf);
    // m_QT.swap(tmp_QT);
    // TAG_ALLOCATION(m_QT);
    // ArrayHandle<Scalar> h_QT(m_QT,
    //                         access_location::host,
    //                         access_mode::overwrite);
    // for (unsigned int i = 0; i < m_QT.size(); i++)
    //     h_QT.data[i] = 0.0;
    // // S initialize
    // GlobalVector<Scalar> tmp_S(max_num_particles, m_exec_conf);
    // m_S.swap(tmp_S);
    // TAG_ALLOCATION(m_S);
    // ArrayHandle<Scalar> h_S(m_S,
    //                         access_location::host,
    //                         access_mode::overwrite);
    // for (unsigned int i = 0; i < m_S.size(); i++)
    //     h_S.data[i] = 0.0;
    
    GlobalVector<Scalar> tmp_c(max_num_particles, m_exec_conf);
    m_c.swap(tmp_c);
    TAG_ALLOCATION(m_c);
    ArrayHandle<Scalar> h_c(m_c,
                            access_location::host,
                            access_mode::overwrite);
    for (unsigned int i = 0; i < m_c.size(); i++)
        h_c.data[i] = 0.0;

    m_kT1 = new Scalar[m_pdata->getNTypes()];
    m_kT2 = new Scalar[m_pdata->getNTypes()];
    m_kH1 = new Scalar[m_pdata->getNTypes()];
    m_kH2 = new Scalar[m_pdata->getNTypes()];
    m_kS1 = new Scalar[m_pdata->getNTypes()];
    m_kS2 = new Scalar[m_pdata->getNTypes()];
    m_Q0 = new Scalar[m_pdata->getNTypes()]; // lower threshold for gamma
    m_Q1 = new Scalar[m_pdata->getNTypes()]; // upper threshold for U
    m_kklino = new Scalar[m_pdata->getNTypes()]; // upper threshold for U
    m_noise_Q = new Scalar[m_pdata->getNTypes()];
    m_U0 = new Scalar[m_pdata->getNTypes()];
    m_U1 = new Scalar[m_pdata->getNTypes()];
    m_gamma0 = new Scalar[m_pdata->getNTypes()];
    m_c0_PHD = new Scalar[m_pdata->getNTypes()];
    m_dc0 = new Scalar[m_pdata->getNTypes()];
    m_sigma_QH = new Scalar[m_pdata->getNTypes()];
    m_sigma_QT = new Scalar[m_pdata->getNTypes()];

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_f_activeVec.get(),
                      sizeof(Scalar4) * m_f_activeVec.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);

        cudaMemAdvise(m_t_activeVec.get(),
                      sizeof(Scalar4) * m_t_activeVec.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);

        // cudaMemAdvise(m_tumble_rate.get(),
        //               sizeof(Scalar) * m_tumble_rate.getNumElements(),
        //               cudaMemAdviseSetReadMostly,
        //               0);
        }
#endif

    }

MixedActiveForceCompute::~MixedActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying MixedActiveForceCompute" << std::endl;

    delete[] m_kT1;
    delete[] m_kT2;
    delete[] m_kH1;
    delete[] m_kH2;
    delete[] m_kS1;
    delete[] m_kS2;
    delete[] m_Q0; // lower threshold for gamma
    delete[] m_Q1; // upper threshold for U
    delete[] m_kklino;
    delete[] m_noise_Q;
    delete[] m_U0;
    delete[] m_U1;
    delete[] m_gamma0;
    delete[] m_c0_PHD;
    delete[] m_dc0;
    delete[] m_sigma_QH;
    delete[] m_sigma_QT;

    m_kT1 = NULL;
    m_kT2 = NULL;
    m_kH1 = NULL;
    m_kH2 = NULL;
    m_kS1 = NULL;
    m_kS2 = NULL;
    m_Q0 = NULL; // lower threshold for gamma
    m_Q1 = NULL; // threshold for taxis
    m_kklino = NULL;
    m_noise_Q = NULL;
    m_U0 = NULL;
    m_U1 = NULL;
    m_gamma0 = NULL;
    m_c0_PHD = NULL;
    m_dc0 = NULL;
    m_sigma_QH = NULL;
    m_sigma_QT = NULL;

    }

void MixedActiveForceCompute::setMixedActiveForce(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar4 f_activeVec;
    f_activeVec.x = pybind11::cast<Scalar>(v[0]);
    f_activeVec.y = pybind11::cast<Scalar>(v[1]);
    f_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar f_activeMag = slow::sqrt(f_activeVec.x * f_activeVec.x + f_activeVec.y * f_activeVec.y
                                    + f_activeVec.z * f_activeVec.z);

    if (f_activeMag > 0)
        {
        f_activeVec.x /= f_activeMag;
        f_activeVec.y /= f_activeMag;
        f_activeVec.z /= f_activeMag;
        f_activeVec.w = f_activeMag;
        }
    else
        {
        f_activeVec.x = 1;
        f_activeVec.y = 0;
        f_activeVec.z = 0;
        f_activeVec.w = 0;
        }

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec,
                                       access_location::host,
                                       access_mode::readwrite);
    h_f_activeVec.data[typ] = f_activeVec;
    }

pybind11::tuple MixedActiveForceCompute::getMixedActiveForce(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec, access_location::host, access_mode::read);

    Scalar4 f_activeVec = h_f_activeVec.data[typ];
    v.append(f_activeVec.w * f_activeVec.x);
    v.append(f_activeVec.w * f_activeVec.y);
    v.append(f_activeVec.w * f_activeVec.z);
    return pybind11::tuple(v);
    }

void MixedActiveForceCompute::setActiveTorque(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar4 t_activeVec;
    t_activeVec.x = pybind11::cast<Scalar>(v[0]);
    t_activeVec.y = pybind11::cast<Scalar>(v[1]);
    t_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar t_activeMag = slow::sqrt(t_activeVec.x * t_activeVec.x + t_activeVec.y * t_activeVec.y
                                    + t_activeVec.z * t_activeVec.z);

    if (t_activeMag > 0)
        {
        t_activeVec.x /= t_activeMag;
        t_activeVec.y /= t_activeMag;
        t_activeVec.z /= t_activeMag;
        t_activeVec.w = t_activeMag;
        }
    else
        {
        t_activeVec.x = 0;
        t_activeVec.y = 0;
        t_activeVec.z = 0;
        t_activeVec.w = 0;
        }

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec,
                                       access_location::host,
                                       access_mode::readwrite);
    h_t_activeVec.data[typ] = t_activeVec;
    }

pybind11::tuple MixedActiveForceCompute::getActiveTorque(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec, access_location::host, access_mode::read);
    Scalar4 t_activeVec = h_t_activeVec.data[typ];
    v.append(t_activeVec.w * t_activeVec.x);
    v.append(t_activeVec.w * t_activeVec.y);
    v.append(t_activeVec.w * t_activeVec.z);
    return pybind11::tuple(v);
    }

// set and get for confidence dynamics params
void MixedActiveForceCompute::setParams(unsigned int type, Scalar kT1,
    Scalar kT2,
    Scalar kH1,
    Scalar kH2,
    Scalar kS1,
    Scalar kS2,
    Scalar Q0, 
    Scalar Q1, 
    Scalar kklino,
    Scalar noise_Q,
    Scalar U0,
    Scalar U1,
    Scalar gamma0,
    Scalar c0_PHD,
    Scalar dc0,
    Scalar sigma_QH,
    Scalar sigma_QT){
    // make sure the type is valid
    if (type >= m_pdata->getNTypes()){
        throw std::invalid_argument("Type does not exist");
    }
    m_kT1[type] = kT1;
    m_kT2[type] = kT2;
    m_kH1[type] = kH1;
    m_kH2[type] = kH2;
    m_kS1[type] = kS1;
    m_kS2[type] = kS2;
    m_Q0[type] =  Q0;
    m_Q1[type] =  Q1;
    m_kklino[type] = kklino;
    m_noise_Q[type] = noise_Q;
    m_U0[type] = U0;
    m_U1[type] = U1;
    m_gamma0[type] = gamma0;
    m_c0_PHD[type] = c0_PHD;
    m_dc0[type] = dc0;
    m_sigma_QH[type] = sigma_QH;
    m_sigma_QT[type] = sigma_QT;
    printf("set params: m_kT1=%g,m_kT2=%g,m_kH1=%g,m_kH2=%g,m_Q0=%g,m_kklino=%g,m_noise_Q=%g,m_U0=%g,m_U1=%g, m_gamma0=%g, m_c0=%g,m_dc0=%g,m_sigma_QT=%g\n", kT1, kT2, kH1, kH2, Q0, kklino, noise_Q, U0, U1, gamma0, c0_PHD, dc0, sigma_QT);
}

void MixedActiveForceCompute::setParamsPython(std::string type, pybind11::dict params){
    auto typ = m_pdata->getTypeByName(type);
    auto _params = mixedactive_params(params);
    setParams(typ, _params.kT1,
    _params.kT2,
    _params.kH1,
    _params.kH2,
    _params.kS1,
    _params.kS2,
    _params.Q0, 
    _params.Q1, 
    _params.kklino,
    _params.noise_Q,
    _params.U0,
    _params.U1,
    _params.gamma0,
    _params.c0_PHD,
    _params.dc0,
    _params.sigma_QH,
    _params.sigma_QT);
    }

pybind11::dict MixedActiveForceCompute::getParams(std::string type){
    auto typ = m_pdata->getTypeByName(type);
    if (typ >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid angle type.");
        }

    pybind11::dict params;
    params["kT1"] = m_kT1[typ];
    params["kT2"] = m_kT2[typ];
    params["kH1"] = m_kH1[typ];
    params["kH2"] = m_kH2[typ];
    params["kS1"] = m_kS1[typ];
    params["kS2"] = m_kS2[typ];
    params["Q0"] = m_Q0[typ];
    params["Q1"] = m_Q1[typ];
    params["kklino"] = m_kklino[typ];
    params["noise_Q"] = m_noise_Q[typ];
    params["U0"] = m_U0[typ];
    params["U1"] = m_U1[typ];
    params["gamma0"] = m_gamma0[typ];
    params["c0_PHD"] = m_c0_PHD[typ];
    params["dc0"] = m_dc0[typ];
    params["sigma_QH"] = m_sigma_QH[typ];
    params["sigma_QT"] = m_sigma_QT[typ];

    return params;
    }



/*! This function sets appropriate active forces on all active particles.
 */
void MixedActiveForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_t_actVec(m_t_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_U(m_U, access_location::host, access_mode::read);
    // 
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar4> h_QS(m_pdata->getConfidences(), access_location::host, access_mode::read);

    // sanity check
    assert(h_f_actVec.data != NULL);
    assert(h_t_actVec.data != NULL);
    assert(h_U.data != NULL);
    assert(h_orientation.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        if(h_QS.data[idx].z>1){
            continue; // if the worm gives up just stop moving
        }

        Scalar U = h_U.data[idx];
        vec3<Scalar> f(h_f_actVec.data[type].w * h_f_actVec.data[type].x,
                       h_f_actVec.data[type].w * h_f_actVec.data[type].y,
                       h_f_actVec.data[type].w * h_f_actVec.data[type].z);
        quat<Scalar> quati(h_orientation.data[idx]);
        vec3<Scalar> fi = rotate(quati, f);
        h_force.data[idx] = vec_to_scalar4(U * fi, 0);

        vec3<Scalar> t(h_t_actVec.data[type].w * h_t_actVec.data[type].x,
                       h_t_actVec.data[type].w * h_t_actVec.data[type].y,
                       h_t_actVec.data[type].w * h_t_actVec.data[type].z);
        vec3<Scalar> ti = rotate(quati, t);
        h_torque.data[idx] = vec_to_scalar4(ti, 0);
        }
    }

/*! This function applies rotational diffusion to the orientations of all active particles. The
 orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
void MixedActiveForceCompute::rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep)
    {
    //  array handles
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_QS(m_pdata->getConfidences(), access_location::host, access_mode::readwrite);

    assert(h_f_actVec.data != NULL);
    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_tag.data != NULL);

    const auto rotation_constant = slow::sqrt(2.0 * rotational_diffusion * m_deltaT);
    // std::cout << "now in rotational diffusion: rotation_constant = " << rotation_constant << std::endl;
    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        if (h_f_actVec.data[type].w != 0 && h_QS.data[idx].z < 1)
            {
            unsigned int ptag = h_tag.data[idx];
            hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::MixedActiveForceCompute,
                                                   timestep,
                                                   m_sysdef->getSeed()),
                                       hoomd::Counter(ptag));

            quat<Scalar> quati(h_orientation.data[idx]);

            if (m_sysdef->getNDimensions() == 2) // 2D
                {
                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);

                vec3<Scalar> b(0, 0, 1.0);
                quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(b, delta_theta);

                quati = rot_quat * quati; // rotational diffusion quaternion applied to orientation
                quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
                h_orientation.data[idx] = quat_to_scalar4(quati);
                // In 2D, the only meaningful torque vector is out of plane and should not change
                }
            else // 3D: Following Stenhammar, Soft Matter, 2014
                {
                hoomd::SpherePointGenerator<Scalar> unit_vec;
                vec3<Scalar> rand_vec;
                unit_vec(rng, rand_vec);

                vec3<Scalar> f(h_f_actVec.data[type].x,
                               h_f_actVec.data[type].y,
                               h_f_actVec.data[type].z);
                vec3<Scalar> fi
                    = rotate(quati, f); // rotate active force vector from local to global frame

                vec3<Scalar> aux_vec = cross(fi, rand_vec); // rotation axis
                Scalar aux_vec_mag = slow::rsqrt(dot(aux_vec, aux_vec));
                aux_vec *= aux_vec_mag;

                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);
                quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(aux_vec, delta_theta);

                quati = rot_quat * quati; // rotational diffusion quaternion applied to orientation
                quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
                h_orientation.data[idx] = quat_to_scalar4(quati);
                }
            }
        }
    }

bool MixedActiveForceCompute::should_tumble(Scalar tumble_rate, Scalar time_elapse, hoomd::RandomGenerator rng){
    // model tumbling as a Poisson process
    Scalar timeForNextEvent = hoomd::GammaDistribution<Scalar>(1, 1/tumble_rate)(rng);
    return timeForNextEvent <= time_elapse;
}

void MixedActiveForceCompute::general_turn(uint64_t period, uint64_t timestep, Scalar tumble_angle_gauss_spread, bool iftaxis){
    // if tumble_angle_gauss_spread<0 no klino kinesis
    //  array handles
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_tumble_rate(m_pdata->getTumbles(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_QS(m_pdata->getConfidences(), access_location::host, access_mode::readwrite);

    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_tumble_rate.data != NULL);
    assert(h_tag.data != NULL);
    assert(h_QS.data != NULL);

    Scalar tmpQ, gamma, c_new, c_old;
    Scalar time_elapse = m_deltaT * period;
    Scalar4 pos;
    unsigned int idx, typ, ptag;

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        idx = m_group->getMemberIndex(i);
        typ = __scalar_as_int(h_pos.data[idx].w);
        ptag = h_tag.data[idx];

        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::MixedActiveForceCompute,
        timestep,m_sysdef->getSeed()),hoomd::Counter(ptag));
        
        tmpQ = h_QS.data[idx].x + h_QS.data[idx].y + hoomd::NormalDistribution<Scalar>(m_noise_Q[typ], 0)(rng);
        pos = h_pos.data[idx];

        if (m_sysdef->getNDimensions() == 2) // 2D
        {
            gamma = h_tumble_rate.data[idx].x;
            // c_new = compute_c_new(pos, timestep);
            // c_old = h_QS.data[idx].w;
            // if(c_new<c_old && tumble_angle_gauss_spread>0){
            //     gamma += gamma * (1+tanh(tmpQ - m_Q0[typ]))/2;
            //     h_tumble_rate.data[idx].x = gamma;
            // }
            // now decide whether to tumble at this timestep
            if(!should_tumble(gamma, time_elapse, rng)){
                continue;
            }
            vec3<Scalar> rot_axis(0.0, 0.0, 1.0);
            
            Scalar theta_tumble;
            if (tumble_angle_gauss_spread<0)
            {
                theta_tumble = hoomd::UniformDistribution<Scalar>(-M_PI, M_PI )(rng);
            }
            else{
                theta_tumble = hoomd::NormalDistribution<Scalar>(tumble_angle_gauss_spread, M_PI)(rng);
                theta_tumble = (theta_tumble>M_PI) ? (M_PI - theta_tumble) : theta_tumble;
            }
            // now theta_tumble is [-pi , pi]
            Scalar theta_turn;
            if(iftaxis){
                Scalar3 cgrad = compute_c_grad(pos, timestep);
                Scalar gradx, grady; 
                gradx = cgrad.x; grady = cgrad.y;
                Scalar theta_taxis = atan2(grady, gradx);
                Scalar cosq, sinq, theta0;
                cosq = h_orientation.data[idx].x;
                sinq = h_orientation.data[idx].w;
                theta0 = atan2(sinq,cosq)*2.0;
                theta_taxis -= theta0;
                // so that the angle to rotate falls in [-2pi, 2pi] 
                Scalar frac_taxis = (tanh(tmpQ-2*m_Q0[typ])+1)/2; // linear mixture of taxis angle and the tumble angle.
                theta_turn = theta_taxis * std::min(frac_taxis,1.0) + theta_tumble * std::max(1.0-frac_taxis, 0.0);
            }
            else{
                theta_turn = theta_tumble;
            }
            quat<Scalar> quati = quat<Scalar>::fromAxisAngle(rot_axis, theta_turn);
            quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
            h_orientation.data[idx] = quat_to_scalar4(quati);
            // In 2D, the only meaningful torque vector is out of plane and should not change
        }
        else // 3D: Following Stenhammar, Soft Matter, 2014
        {
            // TODO: implement 3D
            
        }
    }
}

Scalar MixedActiveForceCompute::update_Q(Scalar &Q, Scalar c_old, Scalar c_new, int FLAG_Q, unsigned int typ){
    Scalar k1, k2, c_term;

    switch (FLAG_Q) {
    case m_FLAG_QH: {
        k1 = m_kH1[typ];
        k2 = m_kH2[typ];
        c_term = (c_new - c_old)/m_deltaT;
        if(c_term<=0){
            c_term = 0.0;
            break;
        }
        // c_term = (c_term>m_dc0[typ]) ? 1.0 : exp(-pow(log(c_term/m_dc0[typ])/m_sigma_QH[typ],2.0));
        c_term = (c_term>m_dc0[typ]) ? 1.0 : (log10(c_term/m_dc0[typ])+3)/3;
        c_term = std::max(0.0, c_term);
        break;
    }
    case m_FLAG_QT: {
        k1 = m_kT1[typ];
        k2 = m_kT2[typ];
        if(c_new<1e-10){
            c_term = 0.0;
            break;
        }
        // from optogenetic exp: when both H&T stimed, goes forward
        c_term = 0.5 * exp(-pow(log(c_new/m_c0_PHD[typ])/m_sigma_QT[typ],2.0));
        break;
    }
    default:
        printf("FLAG_Q must be either for QH or QT!\n");
        return -1;
    } 

    Q += m_deltaT * ((-k1) * Q + k2 * c_term);
    return c_term;
}

void MixedActiveForceCompute::update_S(Scalar &S, Scalar Q, unsigned int typ){
    Scalar k1, k2;
    k1 = m_kS1[typ];
    k2 = m_kS2[typ];
    S -= m_deltaT*(k1 * S + k2*(Q-0.3));
}

void MixedActiveForceCompute::update_U(Scalar &U, Scalar Q, unsigned int typ){
    Scalar U0, U1, Q0;
    U0 = m_U0[typ];
    U1 = m_U1[typ];
    Q0 = m_Q0[typ];
    U = U0 + U1 * tanh(Q) / tanh(Q0); 
    // QT should saturate to 1 to make it more symmetric around mean U0. 
    // If QH=0, QT=Q0, U=U0-U1.
    // if QH = QT = Q0, U=U0. 
    // if QH=Q0, QT=0, U=U0+U1.
    // if QH=1+Q0, QT=1, U=U0+U1.
}

void MixedActiveForceCompute::update_U_random(Scalar &U, unsigned int typ, unsigned int ptag){
    Scalar U0, U1;
    U0 = m_U0[typ];
    U1 = m_U1[typ];
    hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::MixedActiveForceCompute,ptag,m_sysdef->getSeed()), hoomd::Counter(ptag));
    U = hoomd::NormalDistribution<Scalar>(U0/3, U0)(rng);
    U = (U>0.0) ? U : 0.0;
}

void MixedActiveForceCompute::update_tumble_rate(Scalar &gamma, Scalar Q, unsigned int typ){
    Scalar Q0, gamma0;
    gamma0 = m_gamma0[typ];
    Q0 = m_Q0[typ];
    // gamma = gamma0 * (1 - tanh(Q-Q0)); 
    gamma = gamma0 * exp(-(Q-Q0)*m_kklino[typ]);
    // so that when Q=0 gamma=gamma0
}

Scalar MixedActiveForceCompute::compute_c_new(Scalar4 pos, uint64_t timestep){
    Scalar x, y, t;
    x = pos.x; y = pos.y;
    t = m_deltaT * timestep;
    return m_grid_data->getData(x, y, t);
}

Scalar3 MixedActiveForceCompute::compute_c_grad(Scalar4 pos, uint64_t timestep){
    Scalar x, y, t;
    x = pos.x; y = pos.y;
    t = m_deltaT * timestep;
    Scalar dcdx, dcdy;
    dcdx = m_grid_data->getGradX(x, y, t);
    dcdy = m_grid_data->getGradY(x, y, t);
    Scalar3 gradc(make_scalar3(dcdx, dcdy, 0.0));
    return gradc;
}

/************ end aux methods for internal confidence calculations ************/


void MixedActiveForceCompute::update_dynamical_parameters(uint64_t timestep){
    //  update the swim speed by rescaling f_actVec;
    //  update the tumble rate;
    //  array handles
    // printf("now at timestep %d: update dynamical parameters\n", timestep);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_tumble_rate(m_pdata->getTumbles(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_QS(m_pdata->getConfidences(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_U(m_U, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    Scalar QH, QT, S, U, gamma, c_old, c_new, cterm; 
    unsigned int idx, typ, ptag;

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
    {
        idx = m_group->getMemberIndex(i);
        typ = __scalar_as_int(h_pos.data[idx].w);
        ptag = h_tag.data[idx];

        S = h_QS.data[idx].z;
        if(S>=1){
            continue;
        }
        QH = h_QS.data[idx].x;
        QT = h_QS.data[idx].y;
        c_old = h_QS.data[idx].w;
        U = h_U.data[idx];
        gamma = h_tumble_rate.data[idx].x;
        Scalar4 pos = h_pos.data[idx];
        // printf("now at timestep %d, part %d : c_old=%g. before compute new c\n", timestep, idx, c_old);
        c_new = compute_c_new(pos, timestep);
        // now evolve the dynamics
        cterm = update_Q(QH, c_old, c_new, m_FLAG_QH, typ);
        h_tumble_rate.data[idx].z = cterm;
        cterm = update_Q(QT, c_old, c_new, m_FLAG_QT, typ);
        h_tumble_rate.data[idx].w = cterm;
        QT = QT > m_Q0[typ] ? m_Q0[typ] : QT; // let tail confidence saturate at Q0. 
        QH = QH > 2*m_Q0[typ] ? 2*m_Q0[typ] : QH; // QH saturate at 2QT
        
        // update_S(S, QH + QT, typ);
        
        if(m_orthokinesis){
            update_U(U, QH-QT, typ);
        }
        else{
            update_U_random(U, typ, ptag);
        }
        
        if(m_klinokinesis){
            update_tumble_rate(gamma, QH-QT, typ);
        }
        else{
            gamma = m_gamma0[typ];
        }
        h_tumble_rate.data[idx].x = gamma;
        h_tumble_rate.data[idx].y = U;
        // now update the device values
        h_QS.data[idx].x = QH;
        h_QS.data[idx].y = QT;
        h_QS.data[idx].z = S;
        h_QS.data[idx].w = c_new;
        h_U.data[idx] = U;
        
    }
    // printf("\n\n");
}

/*! This function updates dynamical parameters and sets forces for all active particles
    \param timestep Current timestep
*/
void MixedActiveForceCompute::computeForces(uint64_t timestep){
    update_dynamical_parameters(timestep);
    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

void MixedActiveForceCompute::setConcentrationFile(const std::string& filename){
    if (!m_grid_data) {
        throw std::runtime_error("Grid data is not initialized.");
    }
    printf("now load concentration data from file...\n");
    m_grid_data->loadDataFromFile(filename);
}

void MixedActiveForceCompute::setGridSize(int Nx, int Ny, int Nt, double xMin, double xMax, double yMin, double yMax, double tMin, double tMax){
    if (!m_grid_data) {
        throw std::runtime_error("Grid data is not initialized.");
    }
    printf("now setting the grid size for m_grid_data. before setting size is %ld\n", m_grid_data->getGridSize());
    m_grid_data->setGridSize(Nx, Ny, Nt, xMin, xMax, yMin, yMax, tMin, tMax);
    printf("now grid size is %ld\n", m_grid_data->getGridSize());
}

namespace detail
    {
void export_MixedActiveForceCompute(pybind11::module& m)
    {
    pybind11::class_<MixedActiveForceCompute, ForceCompute, std::shared_ptr<MixedActiveForceCompute>>(
        m,
        "MixedActiveForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, bool, bool>())
        .def("setMixedActiveForce", &MixedActiveForceCompute::setMixedActiveForce)
        .def("getMixedActiveForce", &MixedActiveForceCompute::getMixedActiveForce)
        .def("setActiveTorque", &MixedActiveForceCompute::setActiveTorque)
        .def("getActiveTorque", &MixedActiveForceCompute::getActiveTorque)
        .def("setParams", &MixedActiveForceCompute::setParamsPython)
        .def("getParams", &MixedActiveForceCompute::getParams)
        .def("setGridSize", &MixedActiveForceCompute::setGridSize)
        .def("setConcentrationFile", &MixedActiveForceCompute::setConcentrationFile)
        // .def("getConfidence", &MixedActiveForceCompute::getConfidencePython)
        .def_property_readonly("filter",
                               [](MixedActiveForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
