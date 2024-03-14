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


class RectGridData{
private:
    std::vector<std::vector<std::vector<double>>> grid; // 3D vector for data values
    std::vector<std::vector<bool>> grid_empty; // 2D vector to mark if grid point has been initialized with c data. 1 if not.
    double xMin, yMin, tMin; // Minimum bounds
    double xMax, yMax, tMax; // Max bounds for
    double deltaX, deltaY, deltaT; // Discretization steps
    int Nx, Ny, Nt; // Sizes of the grid in each dimension
    std::vector<double> xcoord;
    std::vector<double> ycoord;
    std::vector<double> tcoord;

public:
    RectGridData(double xMin=-30, double xMax=30, double yMin=-30, double yMax=30, double tMin=0, double tMax=600, double deltaX=0.5, double deltaY=0.5, double deltaT=10): xMin(xMin), xMax(xMax), yMin(yMin), yMax(yMax), tMin(tMin), tMax(tMax), deltaX(deltaX), deltaY(deltaY), deltaT(deltaT){
        Nx = static_cast<int> ((xMax - xMin)/deltaX)+1;
        Ny = static_cast<int> ((yMax - yMin)/deltaY)+1;
        Nt = static_cast<int> ((tMax - tMin)/deltaT)+1;
        grid.resize(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nt, 0.0)));
        grid_empty.resize(Nx, std::vector<bool>(Ny, true));
    }
    void setGridSize(int Nx, int Ny, int Nt, double xMin, double xMax, double yMin, double yMax, double tMin, double tMax){
        // Nx, Ny, Nt are the number of points
        printf("now within RectGridData setGridSize: Nx=%d,Ny=%d,Nt=%d\n", Nx, Ny,Nt);
        deltaX = (xMax-xMin)/(Nx-1); 
        deltaY = (yMax-yMin)/(Ny-1);
        deltaT = (tMax-tMin)/(Nt-1);
        Nx = Nx;
        Ny = Ny;
        Nt = Nt;
        grid.resize(Nx, std::vector<std::vector<double>>(Ny, std::vector<double>(Nt, 0.0)));
        grid_empty.resize(Nx, std::vector<bool>(Ny, true));
    }
    unsigned long int getGridSize(){
        return grid.size();
    }
    void setData(int indx, int indy, int indt, double value) {
        if (indx < 0 || indx >= Nx || indy < 0 || indy >= Ny || indt < 0 || indt >= Nt) {
            printf("xMin=%g, xMax=%g, dx=%g, y min=%g, max=%g, dy=%g, Nx=%d, Ny=%d\n", xMin, xMax, deltaX, yMin, yMax, deltaY, Nx, Ny);
            throw std::out_of_range("Data point coordinates out of grid bounds.");
        }
        grid[indx][indy][indt] = value;
        grid_empty[indx][indy] = false;
    }

    double getData(double x, double y, double t) {
        // Convert to grid indices
        double ix = (x - xMin) / deltaX;
        double iy = (y - yMin) / deltaY;
        double it = (t - tMin) / deltaT;

        // Find the bounding indices for interpolation
        int ixLow = std::max(0, static_cast<int>(floor(ix)));
        int ixHigh = std::min(Nx - 1, ixLow + 1);
        int iyLow = static_cast<int>(floor(iy)) % Ny;
        int iyHigh = (iyLow + 1) % Ny;
        int itLow = std::max(0, static_cast<int>(floor(it)));
        int itHigh = std::min(Nt - 1, itLow + 1);

        // Compute weights for interpolation
        double wx = ix - ixLow;
        double wy = iy - iyLow;
        double wT = it - itLow;

        // Interpolate along r, y, and t
        double value = 0.0;
        for (int i : {ixLow, ixHigh}) {
            for (int j : {iyLow, iyHigh}) {
                for (int k : {itLow, itHigh}) {
                    double weight = (i == ixLow ? 1 - wx : wx) *
                                    (j == iyLow ? 1 - wy : wy) *
                                    (k == itLow ? 1 - wT : wT);
                    value += grid[i][j][k] * weight;
                }
            }
        }
        return value;
    }
    void show_unfilled_grid(){
        for (int i = 0; i < Nx; i++)
        {
            for (int j = 0; j < Ny; j++)
            {
                if(grid_empty[i][j]){
                    printf("grid point @x=%g,y=%g is not initialized\n", xcoord[i], ycoord[j]);
                }
            }
            
        }
        
    }
    void loadDataFromFile(const std::string& filename) {
        std::ifstream file(filename);
        bool isOpen = file.is_open();
        std::cout << "File open: " << isOpen << std::endl;

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        // Skip metadata lines
        while (std::getline(file, line)) {
            if (line.empty() || line[0] != '%') {
                break; // Reached the data section
            }
        }

        // Continue with the first line of data
        int nlines = 0;
        int indx, indy, indt;
        indx = 0; indt = 0;
        do {
            if (line.empty() || line[0] == '%') continue; // Skip any potential empty line or late metadata
            std::istringstream iss(line);
            double value;
            indy = 0;
            while (iss >> value) {
                switch (nlines)
                {
                case 0:
                    // 1st line is x coords
                    xcoord.push_back(value);
                    break;
                case 1:
                    // 2nd line is y coords
                    ycoord.push_back(value);
                    break;
                case 2:
                    // 3rd line is t coords
                    tcoord.push_back(value);
                    break;
                default:
                    // find current index
                    setData(indx,indy,indt,value);
                    indy++;
                    if(indy==Ny){
                        indy = 0;
                        indx++;
                        if(indx==Nx){
                            indx = 0;
                            indt++;
                        }
                    }
                    break;
                }
            }
            if (nlines==0){
                std::cout << "read xcoords: ";
                for (const double& value : xcoord) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }
            if (nlines==1){
                std::cout << "read ycoords: ";
                for (const double& value : ycoord) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }
            if (nlines==2){
                std::cout << "read tcoords: ";
                for (const double& value : tcoord) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            }
            if (nlines==2)
            {
                // just finished reading x, y, t coords.
                int Nx, Ny, Nt;
                Nx = xcoord.size();
                Ny = ycoord.size();
                Nt = tcoord.size();
                xMin = xcoord[0]; xMax = xcoord.back();
                yMin = ycoord[0]; yMax = ycoord.back();
                tMin = tcoord[0]; tMax = tcoord.back();
                printf("now just finished reading c file. set grid size: Nx=%d,Ny=%d,Nt=%d, xmin=%g,xmax=%g,ymin=%g,ymax=%g,tmin=%g,tmax=%g\n", Nx, Ny, Nt, xMin, xMax, yMin, yMax, tMin, tMax);
                setGridSize(Nx, Ny, Nt, xMin, xMax, yMin, yMax, tMin, tMax);
            }
            nlines+=1;
        } while (std::getline(file, line));

        file.close();
        // interpolate();
        show_unfilled_grid();
    }
};

struct SurroundingPointIndex {
    int i; // radial index
    int j; // angular index
};
class PolarDataGrid {
private:
    std::vector<std::vector<std::vector<double>>> grid; // 3D vector for data values
    std::vector<std::vector<bool>> grid_empty; // 2D vector to mark if grid point has been initialized with c data. 1 if not.
    double rMin, thetaMin, tMin; // Minimum bounds for r, theta, and t
    double rMax, thetaMax, tMax; // Max bounds for r, theta, and t
    double deltaR, deltaTheta, deltaT; // Discretization steps for r, theta, and t
    int rSize, thetaSize, Nt; // Sizes of the grid in each dimension

public:
    PolarDataGrid(double rMin = 0, double rMax = 10, double thetaMin = 0, 
                double thetaMax = 2 * M_PI,
                double tMin = 0, double tMax = 600, double deltaR = 0.1, double deltaTheta = M_PI / 180,
                double deltaT = 10)
        : rMin(rMin), thetaMin(thetaMin), tMin(tMin), rMax(rMax), thetaMax(thetaMax), tMax(tMax), deltaR(deltaR), deltaTheta(deltaTheta), deltaT(deltaT) {
        rSize = static_cast<int>((rMax - rMin) / deltaR) + 1;
        thetaSize = static_cast<int>((thetaMax - thetaMin) / deltaTheta) + 1;
        Nt = static_cast<int>((tMax - tMin) / deltaT) + 1;
        grid.resize(rSize, std::vector<std::vector<double>>(thetaSize, std::vector<double>(Nt, 0.0)));
        grid_empty.resize(rSize, std::vector<bool>(thetaSize, true));
    }

    void setGridSize(double dr, double dtheta, double rmax){
        rMax = rmax;
        int Nr = static_cast<int>((rMax - rMin) / dr) + 1;
        int Ntheta = static_cast<int>((thetaMax - thetaMin) / dtheta) + 1;
        printf("now within PolarDataGrid setGridSize: Nr=%d,Ntheta=%d\n", Nr, Ntheta);
        deltaR = dr; 
        deltaTheta = dtheta;
        rSize = Nr;
        thetaSize = Ntheta;
        grid.resize(Nr, std::vector<std::vector<double>>(Ntheta, std::vector<double>(Nt, 0.0)));
        grid_empty.resize(Nr, std::vector<bool>(Ntheta, true));
    }

    unsigned long int getGridSize(){
        return grid.size();
    }

    void setData(double r, double theta, double t, double value) {
        int i = static_cast<int>((r - rMin) / deltaR);
        int j = static_cast<int>((theta - thetaMin) / deltaTheta);
        int k = static_cast<int>((t - tMin) / deltaT);

        if (i < 0 || i >= rSize || j < 0 || j >= thetaSize || k < 0 || k >= Nt) {
            printf("rMin=%g, rMax=%g, dr=%g, theta min=%g, max=%g, dtheta=%g, rSize=%d, thetaSize=%d\n", rMin, rMax, deltaR, thetaMin, thetaMax, deltaTheta, rSize, thetaSize);
            throw std::out_of_range("Data point coordinates out of grid bounds.");
        }
        grid[i][j][k] = value;
        grid_empty[i][j] = false;
    }

    double getData(double r, double theta, double t) {
        // Normalize theta to [0, 2*pi)
        theta = fmod(theta, 2 * M_PI);
        if (theta < 0) theta += 2 * M_PI;

        // Convert to grid indices
        double ir = (r - rMin) / deltaR;
        double itheta = (theta - thetaMin) / deltaTheta;
        double it = (t - tMin) / deltaT;

        // Find the bounding indices for interpolation
        int irLow = std::max(0, static_cast<int>(floor(ir)));
        int irHigh = std::min(rSize - 1, irLow + 1);
        int ithetaLow = static_cast<int>(floor(itheta)) % thetaSize;
        int ithetaHigh = (ithetaLow + 1) % thetaSize;
        int itLow = std::max(0, static_cast<int>(floor(it)));
        int itHigh = std::min(Nt - 1, itLow + 1);

        // Compute weights for interpolation
        double wR = ir - irLow;
        double wTheta = itheta - ithetaLow;
        double wT = it - itLow;

        // Interpolate along r, theta, and t
        double value = 0.0;
        for (int i : {irLow, irHigh}) {
            for (int j : {ithetaLow, ithetaHigh}) {
                for (int k : {itLow, itHigh}) {
                    double weight = (i == irLow ? 1 - wR : wR) *
                                    (j == ithetaLow ? 1 - wTheta : wTheta) *
                                    (k == itLow ? 1 - wT : wT);
                    value += grid[i][j][k] * weight;
                }
            }
        }
        return value;
    }

    void loadDataFromFile(const std::string& filename) {
        std::ifstream file(filename);
        bool isOpen = file.is_open();
        std::cout << "File open: " << isOpen << std::endl;

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        // Skip metadata lines
        while (std::getline(file, line)) {
            if (line.empty() || line[0] != '%') {
                break; // Reached the data section
            }
        }

        // Continue with the first line of data
        do {
            if (line.empty() || line[0] == '%') continue; // Skip any potential empty line or late metadata

            std::istringstream iss(line);
            double x, y, z;
            iss >> x >> y >> z; // Read coordinates

            // Convert (x, y) to polar coordinates (r, theta)
            double r = sqrt(x * x + y * y);
            double theta = atan2(y, x);
            if (theta < 0) theta += 2 * M_PI; // Ensure theta is within [0, 2*pi)

            // Read and set concentration values for each time point
            double value;
            int time = 0; // Assuming time starts at 0 and increments in fixed intervals as per the file format
            while (iss >> value) {
                if(r<2 && time==100){
                    printf("now reading c: @r=%g,theta=%g,time=100: c=%g\n", r, theta, value);
                }
                this->setData(r, theta, time, value);
                time += 10; // Increment time based on the assumption each column is 10 units apart
            }
        } while (std::getline(file, line));

        file.close();
        // interpolate();
        show_unfilled_grid();
    }

    void show_unfilled_grid(){
        for (int i = 0; i < rSize; i++)
        {
            for (int j = 0; j < thetaSize; j++)
            {
                if(grid_empty[i][j]){
                    double r, theta;
                    r = deltaR * i;
                    theta = deltaTheta * j;
                    printf("grid point @r=%g,theta=%g is not initialized\n", r, theta);
                }
            }
            
        }
        
    }

    bool needsInterpolation(int i, int j) {
        return grid_empty[i][j];
    }

};


#if HOOMD_LONGREAL_SIZE == 32
struct __attribute__((aligned(8))) mixedactive_params {
#else
struct __attribute__((aligned(16))) mixedactive_params {
#endif
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
        v["kT1"] = kT1;
        v["kT2"] = kT2;
        v["kH1"] = kH1;
        v["kH2"] = kH2;
        v["kS1"] = kS1;
        v["kS2"] = kS2;
        v["Q0"] = Q0; 
        v["Q1"] = Q1;
        v["noie_Q"] = noise_Q;
        v["U0"] = U0;
        v["U1"] = U1;
        v["gamma0"] = gamma0;
        v["c0_PHD"] = c0_PHD;
        return v;
        }
#endif
};

//! Adds an active force to a number of particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT MixedActiveForceCompute : public ForceCompute{
    public:
    //! Constructs the compute
    MixedActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group, Scalar L);

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

    void setConcentrationFile(const std::string& filename);
    void setGridSize(int Nx, int Ny, int Nt, double xMin, double xMax, double yMin, double yMax, double tMin, double tMax);


    std::shared_ptr<ParticleGroup>& getGroup(){
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
    virtual void update_dynamical_parameters(uint64_t timestep);

    void update_Q(Scalar &Q, Scalar c_new, Scalar c_old, int FLAG_Q, unsigned int typ);
    void update_S(Scalar &S, Scalar gamma, unsigned int typ);
    void update_U(Scalar &U, Scalar Q, unsigned int typ);
    void update_tumble_rate(Scalar &gamma, Scalar Q, unsigned int typ);
    Scalar compute_c_new(Scalar4 pos, uint64_t timestep);

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

    std::unique_ptr<RectGridData> m_grid_data;

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
