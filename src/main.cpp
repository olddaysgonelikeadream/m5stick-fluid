/**
 * FloatingBottle - M5Stack StickS3
 * Full port of the FLIP fluid simulation algorithm.
 *
 * Key implementation notes:
 *   - Particle radius r = 0.35 * h, initialized in hexagonal close-packed layout
 *   - Particle spatial hash spacing: p_inv_spacing = 1 / (2.2 * r)
 *   - Rendering uses the particle density field with gamma correction (not cell type)
 *   - Gravity input is normalized to [-1, 1] and scaled by GRAVITY_SCALE
 *   - CPU-driven sloshing is added as a sinusoidal perturbation on top of IMU gravity
 */

#include <M5Unified.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Screen and grid dimensions
// ---------------------------------------------------------------------------
#define SCREEN_W  135
#define SCREEN_H  240
#define CELL_SIZE 5
#define GRID_W    27   // SCREEN_W / CELL_SIZE
#define GRID_H    48   // SCREEN_H / CELL_SIZE

// Simulation grid includes a one-cell solid border on each side
#define SIM_W  (GRID_W + 2)
#define SIM_H  (GRID_H + 2)
#define NC     (SIM_W * SIM_H)

// ---------------------------------------------------------------------------
// Rendering parameters
// ---------------------------------------------------------------------------
#define LED_VAL_MAX_F  20.0f   // Maximum brightness value used for color mapping
#define DENSITY_CLAMP  1.2f    // Density values above this are clamped to full brightness
#define GAMMA_F        0.6f    // Gamma exponent for perceptual brightness correction
#define COLOR_BG       0x0000  // Background color (black)

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
#define FILL_RATIO     0.55f         // Initial fluid fill level as a fraction of tank height
#define GRAVITY_SCALE  7.0f          // Multiplier applied to normalized IMU gravity
#define FLIP_RATIO     0.9f          // Blend between PIC (0) and FLIP (1); higher = more detail but noisier
#define PUSH_ITERS     1             // Iterations for particle separation pass
#define PRES_ITERS     8             // Gauss-Seidel iterations for pressure solve
#define OVER_RELAX     1.3f          // Over-relaxation factor to accelerate pressure convergence
#define SIM_FPS        30            // Target simulation and render rate
#define DT             (1.0f / 60.0f) // Fixed timestep; smaller than frame interval for stability

// ---------------------------------------------------------------------------
// Gamma lookup table
// Precomputed to avoid calling powf() every frame during rendering.
// ---------------------------------------------------------------------------
static uint8_t s_gamma_lut[256];
static void gamma_init() {
    for (int i = 0; i < 256; i++) {
        float x = i / 255.0f;
        s_gamma_lut[i] = (uint8_t)lrintf(powf(x, GAMMA_F) * 255.0f);
    }
}

// ---------------------------------------------------------------------------
// Cell type constants
// ---------------------------------------------------------------------------
#define AIR_CELL   0
#define FLUID_CELL 1
#define SOLID_CELL 2

// ---------------------------------------------------------------------------
// FlipFluid: core simulation data structure
//
// The MAC (Marker-And-Cell) grid stores velocity components at cell faces:
//   u[i*ny+j] = x-velocity at the left face of cell (i, j)
//   v[i*ny+j] = y-velocity at the bottom face of cell (i, j)
//
// Particle positions and velocities are stored as interleaved arrays:
//   particle_pos[2*i+0], particle_pos[2*i+1] = (x, y) of particle i
//
// The spatial hash grid (p_num_x * p_num_y) is used to accelerate
// neighbor lookups during the particle separation pass.
// ---------------------------------------------------------------------------
struct FlipFluid {
    float density;
    int f_num_x, f_num_y, f_num_cells;
    float h, f_inv_spacing;

    float *u, *v;           // MAC grid velocities (x and y components)
    float *du, *dv;         // Weight accumulators for P-to-G transfer
    float *prev_u, *prev_v; // Grid velocities from previous step (for FLIP correction)
    float *p;               // Pressure field
    float *s;               // Cell solidity: 0 = solid boundary, 1 = open

    int32_t *cell_type;     // Per-cell type: AIR_CELL, FLUID_CELL, or SOLID_CELL

    int max_particles, num_particles, base_particles;
    float *particle_pos;            // Particle positions (x, y interleaved)
    float *particle_vel;            // Particle velocities (vx, vy interleaved)
    float *particle_density;        // Particle density field (grid-resolution)
    float particle_rest_density;    // Reference density, computed from initial state
    float particle_radius;

    // Spatial hash for particle neighbor search
    float p_inv_spacing;
    int p_num_x, p_num_y, p_num_cells;
    int32_t *num_cell_particles;    // Number of particles per hash cell
    int32_t *first_cell_particle;   // Start index in cell_particle_ids per cell
    int32_t *cell_particle_ids;     // Sorted particle ID array
};

static FlipFluid* s_fluid = nullptr;

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
static inline int   cl_i(int x, int lo, int hi)      { return x<lo?lo:x>hi?hi:x; }
static inline float cl_f(float x, float lo, float hi) { return x<lo?lo:x>hi?hi:x; }
static inline float rand01() { return (float)rand()/(float)RAND_MAX; }

// ---------------------------------------------------------------------------
// flip_create: allocate and initialize the fluid simulation
//
// Particles are placed in a hexagonal close-packed (HCP) grid covering
// the lower fill_ratio fraction of the tank. The solid boundary cells
// are set by marking the outermost ring of s[] as 0.
// ---------------------------------------------------------------------------
FlipFluid* flip_create(int vis_w, int vis_h, float fill_ratio) {
    // Map the pixel grid to a unit-width simulation space with square cells
    float sim_w = 1.0f;
    float sim_h = sim_w * (float)(vis_h + 1) / (float)(vis_w + 1);

    int nx = vis_w + 2, ny = vis_h + 2;
    float hx = sim_w / (nx - 1), hy = sim_h / (ny - 1);
    float h = hx < hy ? hx : hy;
    if (!(h > 0)) return nullptr;

    float tank_w = h * (nx - 1);
    float tank_h = h * (ny - 1);

    // Hexagonal close-packed particle layout
    float r  = 0.35f * h;
    float dx = 2.0f * r;
    float dy = sqrtf(3.0f) / 2.0f * dx;
    float rel_w = 0.8f, rel_h = fill_ratio;
    int num_x = (int)floorf((rel_w * tank_w - 2*h - 2*r) / dx);
    int num_y = (int)floorf((rel_h * tank_h - 2*h - 2*r) / dy);
    if (num_x < 1) num_x = 1;
    if (num_y < 1) num_y = 1;
    int base = num_x * num_y;
    int maxp = base + 256 > base * 2 ? base + 256 : base * 2;

    FlipFluid* f = (FlipFluid*)calloc(1, sizeof(FlipFluid));
    if (!f) return nullptr;

    f->density       = 1000.0f;
    f->f_num_x       = nx;
    f->f_num_y       = ny;
    f->h             = h;
    f->f_inv_spacing = 1.0f / h;
    f->f_num_cells   = nx * ny;
    f->particle_radius = r;
    // Spatial hash cell size is 2.2 * r so each cell covers a diameter plus margin
    f->p_inv_spacing = 1.0f / (2.2f * r);
    f->p_num_x       = (int)floorf(tank_w * f->p_inv_spacing) + 1;
    f->p_num_y       = (int)floorf(tank_h * f->p_inv_spacing) + 1;
    f->p_num_cells   = f->p_num_x * f->p_num_y;
    f->max_particles = maxp;
    f->base_particles= base;
    f->particle_rest_density = 0.0f; // Computed on first update_density call

#define ALLOC_F(ptr, cnt) { (ptr)=(float*)calloc((cnt),sizeof(float)); if(!(ptr)){free(f);return nullptr;} }
#define ALLOC_I(ptr, cnt) { (ptr)=(int32_t*)calloc((cnt),sizeof(int32_t)); if(!(ptr)){free(f);return nullptr;} }
    ALLOC_F(f->u,               f->f_num_cells)
    ALLOC_F(f->v,               f->f_num_cells)
    ALLOC_F(f->du,              f->f_num_cells)
    ALLOC_F(f->dv,              f->f_num_cells)
    ALLOC_F(f->prev_u,          f->f_num_cells)
    ALLOC_F(f->prev_v,          f->f_num_cells)
    ALLOC_F(f->p,               f->f_num_cells)
    ALLOC_F(f->s,               f->f_num_cells)
    ALLOC_F(f->particle_density,f->f_num_cells)
    ALLOC_F(f->particle_pos,    2 * maxp)
    ALLOC_F(f->particle_vel,    2 * maxp)
    ALLOC_I(f->cell_type,       f->f_num_cells)
    ALLOC_I(f->num_cell_particles,  f->p_num_cells)
    ALLOC_I(f->first_cell_particle, f->p_num_cells + 1)
    ALLOC_I(f->cell_particle_ids,   maxp)
#undef ALLOC_F
#undef ALLOC_I

    // Place particles in HCP grid; odd rows are offset by r to form the hex pattern
    f->num_particles = base;
    int pidx = 0;
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            f->particle_pos[pidx+0] = h + r + dx*i + ((j%2==0)?0:r);
            f->particle_pos[pidx+1] = h + r + dy*j;
            pidx += 2;
        }
    }

    // Mark boundary cells as solid (s = 0), interior cells as open (s = 1)
    int n = ny;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            f->s[i*n+j] = (i==0||i==nx-1||j==0||j==ny-1) ? 0.0f : 1.0f;

    return f;
}

void flip_destroy(FlipFluid* f) {
    if (!f) return;
    free(f->u); free(f->v); free(f->du); free(f->dv);
    free(f->prev_u); free(f->prev_v); free(f->p); free(f->s);
    free(f->cell_type); free(f->particle_pos); free(f->particle_vel);
    free(f->particle_density); free(f->num_cell_particles);
    free(f->first_cell_particle); free(f->cell_particle_ids);
    free(f);
}

// ---------------------------------------------------------------------------
// integrate_particles: apply gravity and advance particle positions (symplectic Euler)
// A small damping factor simulates viscosity and prevents energy build-up.
// ---------------------------------------------------------------------------
static void integrate_particles(FlipFluid* f, float dt, float gx, float gy) {
    const float damp = 0.995f;
    for (int i = 0; i < f->num_particles; i++) {
        f->particle_vel[2*i+0] = (f->particle_vel[2*i+0] + gx * dt) * damp;
        f->particle_vel[2*i+1] = (f->particle_vel[2*i+1] + gy * dt) * damp;
        f->particle_pos[2*i+0] += f->particle_vel[2*i+0] * dt;
        f->particle_pos[2*i+1] += f->particle_vel[2*i+1] * dt;
    }
}

// ---------------------------------------------------------------------------
// push_particles_apart: resolve particle overlaps using a spatial hash.
//
// Algorithm:
//   1. Build the spatial hash (count sort into cells).
//   2. For each particle, iterate over its 3x3 neighboring hash cells.
//   3. If two particles overlap (distance < 2r), push each outward by half
//      the overlap distance along the separation vector.
// ---------------------------------------------------------------------------
static void push_particles_apart(FlipFluid* f) {
    int pc = f->p_num_cells;
    memset(f->num_cell_particles, 0, pc * sizeof(int32_t));

    // Count particles per cell
    for (int i = 0; i < f->num_particles; i++) {
        int xi = cl_i((int)(f->particle_pos[2*i+0]*f->p_inv_spacing), 0, f->p_num_x-1);
        int yi = cl_i((int)(f->particle_pos[2*i+1]*f->p_inv_spacing), 0, f->p_num_y-1);
        f->num_cell_particles[xi*f->p_num_y+yi]++;
    }

    // Build prefix sums to get start indices (counting sort prefix)
    int first = 0;
    for (int i = 0; i < pc; i++) { first += f->num_cell_particles[i]; f->first_cell_particle[i]=first; }
    f->first_cell_particle[pc] = first;

    // Fill sorted particle ID array (reverse insertion to match prefix layout)
    for (int i = 0; i < f->num_particles; i++) {
        int xi = cl_i((int)(f->particle_pos[2*i+0]*f->p_inv_spacing), 0, f->p_num_x-1);
        int yi = cl_i((int)(f->particle_pos[2*i+1]*f->p_inv_spacing), 0, f->p_num_y-1);
        int cell = xi*f->p_num_y+yi;
        f->first_cell_particle[cell]--;
        f->cell_particle_ids[f->first_cell_particle[cell]] = i;
    }

    // Resolve overlaps
    float md = 2.0f * f->particle_radius, md2 = md*md;
    for (int it = 0; it < PUSH_ITERS; it++) {
        for (int i = 0; i < f->num_particles; i++) {
            float px = f->particle_pos[2*i+0], py = f->particle_pos[2*i+1];
            int pxi=(int)(px*f->p_inv_spacing), pyi=(int)(py*f->p_inv_spacing);
            int x0=pxi>0?pxi-1:0, y0=pyi>0?pyi-1:0;
            int x1=pxi<f->p_num_x-1?pxi+1:f->p_num_x-1;
            int y1=pyi<f->p_num_y-1?pyi+1:f->p_num_y-1;
            for (int xi=x0;xi<=x1;xi++) for (int yi=y0;yi<=y1;yi++) {
                int cell=xi*f->p_num_y+yi;
                for (int k=f->first_cell_particle[cell];k<f->first_cell_particle[cell+1];k++) {
                    int id=f->cell_particle_ids[k];
                    if (id==i) continue;
                    float dx=f->particle_pos[2*id+0]-px, dy=f->particle_pos[2*id+1]-py;
                    float d2=dx*dx+dy*dy;
                    if (d2>md2||d2==0) continue;
                    float d=sqrtf(d2), s=(0.5f*(md-d))/d;
                    dx*=s; dy*=s;
                    f->particle_pos[2*i+0]-=dx; f->particle_pos[2*i+1]-=dy;
                    f->particle_pos[2*id+0]+=dx; f->particle_pos[2*id+1]+=dy;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// handle_collisions: clamp particles to the interior of the tank.
// Velocity is reflected and damped on wall contact (WALL_DAMP = 0.2).
// ---------------------------------------------------------------------------
static void handle_collisions(FlipFluid* f) {
    float h=f->h, r=f->particle_radius;
    float mnx=h+r, mxx=(f->f_num_x-1)*h-r;
    float mny=h+r, mxy=(f->f_num_y-1)*h-r;
    for (int i=0;i<f->num_particles;i++) {
        float x=f->particle_pos[2*i+0], y=f->particle_pos[2*i+1];
        constexpr float WALL_DAMP = 0.20f;
        if(x<mnx){x=mnx;f->particle_vel[2*i+0] *= -WALL_DAMP;}
        if(x>mxx){x=mxx;f->particle_vel[2*i+0] *= -WALL_DAMP;}
        if(y<mny){y=mny;f->particle_vel[2*i+1] *= -WALL_DAMP;}
        if(y>mxy){y=mxy;f->particle_vel[2*i+1] *= -WALL_DAMP;}
        f->particle_pos[2*i+0]=x; f->particle_pos[2*i+1]=y;
    }
}

// ---------------------------------------------------------------------------
// update_density: splat each particle onto the grid using bilinear weights.
// On the first call (rest density == 0), the average density over all fluid
// cells is recorded as the reference rest density for subsequent frames.
// ---------------------------------------------------------------------------
static void update_density(FlipFluid* f) {
    int n=f->f_num_y; float h=f->h, h2=0.5f*h, inv=f->f_inv_spacing;
    memset(f->particle_density, 0, f->f_num_cells*sizeof(float));
    for (int i=0;i<f->num_particles;i++) {
        float x=cl_f(f->particle_pos[2*i+0],h,(f->f_num_x-1)*h);
        float y=cl_f(f->particle_pos[2*i+1],h,(f->f_num_y-1)*h);
        int x0=(int)((x-h2)*inv); float tx=(x-h2-x0*h)*inv; int x1=x0+1<f->f_num_x-1?x0+1:f->f_num_x-2;
        int y0=(int)((y-h2)*inv); float ty=(y-h2-y0*h)*inv; int y1=y0+1<f->f_num_y-1?y0+1:f->f_num_y-2;
        float sx=1-tx, sy=1-ty;
        f->particle_density[x0*n+y0]+=sx*sy; f->particle_density[x1*n+y0]+=tx*sy;
        f->particle_density[x1*n+y1]+=tx*ty; f->particle_density[x0*n+y1]+=sx*ty;
    }
    if (f->particle_rest_density == 0.0f) {
        float sum=0; int cnt=0;
        for(int i=0;i<f->f_num_cells;i++) if(f->cell_type[i]==FLUID_CELL){sum+=f->particle_density[i];cnt++;}
        if(cnt>0) f->particle_rest_density=sum/cnt;
    }
}

// ---------------------------------------------------------------------------
// transfer_vel: bidirectional velocity transfer between particles and MAC grid.
//
// toGrid = true  (P2G): scatter particle velocities onto the grid using
//                        bilinear weights; normalize by accumulated weights.
//                        Also classifies cells as FLUID or AIR.
//                        Solid cell velocities are restored from prev_u/prev_v.
//
// toGrid = false (G2P): gather updated grid velocities back to particles using
//                        the FLIP/PIC blend:
//                        vel = (1 - FLIP_RATIO) * picVel
//                            + FLIP_RATIO * (oldVel + gridDelta)
//                        Only samples from non-air cell faces to avoid
//                        contaminating fluid velocities with air.
// ---------------------------------------------------------------------------
static void transfer_vel(FlipFluid* f, bool toGrid) {
    int n=f->f_num_y; float h=f->h, h2=0.5f*h, inv=f->f_inv_spacing;
    int nc=f->f_num_cells;
    if (toGrid) {
        memcpy(f->prev_u,f->u,nc*sizeof(float)); memcpy(f->prev_v,f->v,nc*sizeof(float));
        memset(f->du,0,nc*sizeof(float)); memset(f->dv,0,nc*sizeof(float));
        memset(f->u,0,nc*sizeof(float)); memset(f->v,0,nc*sizeof(float));
        for(int i=0;i<nc;i++) f->cell_type[i]=(f->s[i]==0)?SOLID_CELL:AIR_CELL;
        for(int i=0;i<f->num_particles;i++){
            int xi=cl_i((int)(f->particle_pos[2*i+0]*inv),0,f->f_num_x-1);
            int yi=cl_i((int)(f->particle_pos[2*i+1]*inv),0,f->f_num_y-1);
            int cell=xi*n+yi;
            if(f->cell_type[cell]==AIR_CELL) f->cell_type[cell]=FLUID_CELL;
        }
    }
    for (int comp=0;comp<2;comp++) {
        float dx=(comp==0)?0:h2, dy=(comp==0)?h2:0;
        float *ff=(comp==0)?f->u:f->v, *pf=(comp==0)?f->prev_u:f->prev_v, *d=(comp==0)?f->du:f->dv;
        for(int i=0;i<f->num_particles;i++){
            float x=cl_f(f->particle_pos[2*i+0],h,(f->f_num_x-1)*h);
            float y=cl_f(f->particle_pos[2*i+1],h,(f->f_num_y-1)*h);
            int x0=(int)((x-dx)*inv); x0=x0<f->f_num_x-2?x0:f->f_num_x-2; float tx=(x-dx-x0*h)*inv;
            int x1=x0+1<f->f_num_x-2?x0+1:f->f_num_x-2;
            int y0=(int)((y-dy)*inv); y0=y0<f->f_num_y-2?y0:f->f_num_y-2; float ty=(y-dy-y0*h)*inv;
            int y1=y0+1<f->f_num_y-2?y0+1:f->f_num_y-2;
            float sx=1-tx,sy=1-ty,w0=sx*sy,w1=tx*sy,w2=tx*ty,w3=sx*ty;
            int r0=x0*n+y0,r1=x1*n+y0,r2=x1*n+y1,r3=x0*n+y1;
            if(toGrid){
                float pv=f->particle_vel[2*i+comp];
                ff[r0]+=pv*w0;d[r0]+=w0;ff[r1]+=pv*w1;d[r1]+=w1;
                ff[r2]+=pv*w2;d[r2]+=w2;ff[r3]+=pv*w3;d[r3]+=w3;
            } else {
                int off=(comp==0)?n:1;
                float v0=(f->cell_type[r0]!=AIR_CELL||f->cell_type[r0-off]!=AIR_CELL)?1:0;
                float v1=(f->cell_type[r1]!=AIR_CELL||f->cell_type[r1-off]!=AIR_CELL)?1:0;
                float v2=(f->cell_type[r2]!=AIR_CELL||f->cell_type[r2-off]!=AIR_CELL)?1:0;
                float v3=(f->cell_type[r3]!=AIR_CELL||f->cell_type[r3-off]!=AIR_CELL)?1:0;
                float vc=f->particle_vel[2*i+comp];
                float ds=v0*w0+v1*w1+v2*w2+v3*w3;
                if(ds>0){
                    float picV=(v0*w0*ff[r0]+v1*w1*ff[r1]+v2*w2*ff[r2]+v3*w3*ff[r3])/ds;
                    float corr=(v0*w0*(ff[r0]-pf[r0])+v1*w1*(ff[r1]-pf[r1])+v2*w2*(ff[r2]-pf[r2])+v3*w3*(ff[r3]-pf[r3]))/ds;
                    f->particle_vel[2*i+comp]=(1-FLIP_RATIO)*picV+FLIP_RATIO*(vc+corr);
                }
            }
        }
        if(toGrid){
            for(int i=0;i<nc;i++) if(d[i]>0) ff[i]/=d[i];
            // Restore solid boundary velocities to prevent fluid from crossing walls
            for(int i=0;i<f->f_num_x;i++) for(int j=0;j<f->f_num_y;j++){
                int idx=i*n+j;
                bool sol=(f->cell_type[idx]==SOLID_CELL);
                if(sol||(i>0&&f->cell_type[(i-1)*n+j]==SOLID_CELL)) f->u[idx]=f->prev_u[idx];
                if(sol||(j>0&&f->cell_type[i*n+j-1]==SOLID_CELL))   f->v[idx]=f->prev_v[idx];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solve_incompressibility: Gauss-Seidel pressure projection.
//
// Iteratively eliminates velocity divergence from each fluid cell by computing
// a pressure correction and adjusting the four face velocities accordingly.
// Density compression is also penalized when local density exceeds rest density.
// ---------------------------------------------------------------------------
static void solve_incompressibility(FlipFluid* f, float dt) {
    int n=f->f_num_y, nc=f->f_num_cells;
    memset(f->p,0,nc*sizeof(float));
    memcpy(f->prev_u,f->u,nc*sizeof(float)); memcpy(f->prev_v,f->v,nc*sizeof(float));
    float cp=f->density*f->h/dt;
    for(int it=0;it<PRES_ITERS;it++){
        for(int i=1;i<f->f_num_x-1;i++) for(int j=1;j<f->f_num_y-1;j++){
            int c=i*n+j;
            if(f->cell_type[c]!=FLUID_CELL) continue;
            int l=(i-1)*n+j,r=(i+1)*n+j,b=i*n+j-1,t=i*n+j+1;
            float sx0=f->s[l],sx1=f->s[r],sy0=f->s[b],sy1=f->s[t],sv=sx0+sx1+sy0+sy1;
            if(sv==0) continue;
            float div=f->u[r]-f->u[c]+f->v[t]-f->v[c];
            if(f->particle_rest_density>0){
                // Extra divergence term to resist local compression above rest density
                float comp=f->particle_density[c]-f->particle_rest_density;
                if(comp>0) div-=comp;
            }
            float pv=-div/sv*OVER_RELAX;
            f->p[c]+=cp*pv;
            f->u[c]-=sx0*pv; f->u[r]+=sx1*pv;
            f->v[c]-=sy0*pv; f->v[t]+=sy1*pv;
        }
    }
}

// ---------------------------------------------------------------------------
// flip_step: advance the simulation by one timestep.
//
// gx, gy: normalized gravity in [-1, 1], sourced from the IMU.
// The full FLIP pipeline runs in this fixed order each frame.
// ---------------------------------------------------------------------------
void flip_step(FlipFluid* f, float dt, float gx, float gy) {
    float Gx = gx * GRAVITY_SCALE;
    float Gy = gy * GRAVITY_SCALE;
    integrate_particles(f, dt, Gx, Gy);   // 1. Apply gravity, move particles
    push_particles_apart(f);               // 2. Resolve particle overlaps
    handle_collisions(f);                  // 3. Enforce boundary constraints
    transfer_vel(f, true);                 // 4. P2G: particles -> grid
    update_density(f);                     // 5. Compute density field
    solve_incompressibility(f, dt);        // 6. Pressure projection
    transfer_vel(f, false);               // 7. G2P: grid -> particles (FLIP blend)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
M5Canvas canvas(&M5.Display);

float g_cpu    = 0.0f;
bool  g_pc_on  = false;
bool  g_cpu_monitor = true;  // Controlled by BtnA; disables serial CPU input when false
uint32_t g_last_rx = 0;
#define PC_TIMEOUT 3000       // ms before CPU monitor resets if no serial data arrives

float g_imu_x = 0.0f;        // Low-pass filtered IMU gravity, normalized to [-1, 1]
float g_imu_y = 0.0f;

// ---------------------------------------------------------------------------
// Heart: a buoyancy-simulated floating object rendered as a heart shape.
// ---------------------------------------------------------------------------
struct Heart {
    float x, y;       // Position in screen pixels
    float vx, vy;     // Velocity in pixels/second
    float angle;      // Rotation angle in radians
    float omega;      // Angular velocity in radians/second
};

static Heart g_heart;
static bool g_heart_visible = true;

void heart_init(FlipFluid* f) {
    // Start near the top of the fluid surface
    g_heart.x     = SCREEN_W * 0.5f;
    g_heart.y     = SCREEN_H * (1.0f - FILL_RATIO) + 20.0f;
    g_heart.vx    = 0.0f;
    g_heart.vy    = 0.0f;
    g_heart.angle = 0.0f;
    g_heart.omega = 0.0f;
}

// ---------------------------------------------------------------------------
// sample_fluid_vel: bilinear interpolation of the MAC grid velocity at a
// given screen-space position. Returns velocity in pixels/second.
// ---------------------------------------------------------------------------
static void sample_fluid_vel(FlipFluid* f, float px, float py, float &out_vx, float &out_vy) {
    // Convert screen pixels to simulation coordinates
    float sim_x = (px / (float)SCREEN_W) * (f->f_num_x - 2) * f->h + f->h;
    float sim_y = (py / (float)SCREEN_H) * (f->f_num_y - 2) * f->h + f->h;

    float inv = f->f_inv_spacing;
    int n = f->f_num_y;

    // Bilinear interpolation of u (x-velocity); sample point offset by -h/2 in y
    float x = sim_x, y = sim_y - 0.5f * f->h;
    int x0 = (int)(x * inv); x0 = x0 < 0 ? 0 : x0 > f->f_num_x-2 ? f->f_num_x-2 : x0;
    int x1 = x0+1 < f->f_num_x-1 ? x0+1 : f->f_num_x-1;
    int y0 = (int)(y * inv); y0 = y0 < 0 ? 0 : y0 > f->f_num_y-2 ? f->f_num_y-2 : y0;
    int y1 = y0+1 < f->f_num_y-1 ? y0+1 : f->f_num_y-1;
    float tx = (x - x0*f->h) * inv, ty = (y - y0*f->h) * inv;
    tx = tx<0?0:tx>1?1:tx; ty = ty<0?0:ty>1?1:ty;
    float u_val = (1-tx)*(1-ty)*f->u[x0*n+y0] + tx*(1-ty)*f->u[x1*n+y0]
                + tx*ty*f->u[x1*n+y1] + (1-tx)*ty*f->u[x0*n+y1];

    // Bilinear interpolation of v (y-velocity); sample point offset by -h/2 in x
    x = sim_x - 0.5f * f->h; y = sim_y;
    x0 = (int)(x * inv); x0 = x0 < 0 ? 0 : x0 > f->f_num_x-2 ? f->f_num_x-2 : x0;
    x1 = x0+1 < f->f_num_x-1 ? x0+1 : f->f_num_x-1;
    y0 = (int)(y * inv); y0 = y0 < 0 ? 0 : y0 > f->f_num_y-2 ? f->f_num_y-2 : y0;
    y1 = y0+1 < f->f_num_y-1 ? y0+1 : f->f_num_y-1;
    tx = (x - x0*f->h) * inv; ty = (y - y0*f->h) * inv;
    tx = tx<0?0:tx>1?1:tx; ty = ty<0?0:ty>1?1:ty;
    float v_val = (1-tx)*(1-ty)*f->v[x0*n+y0] + tx*(1-ty)*f->v[x1*n+y0]
                + tx*ty*f->v[x1*n+y1] + (1-tx)*ty*f->v[x0*n+y1];

    // Scale from simulation units to screen pixels/second
    float scale = (float)SCREEN_W / ((f->f_num_x - 2) * f->h);
    out_vx = u_val * scale;
    out_vy = v_val * scale;
}

// ---------------------------------------------------------------------------
// heart_update: physics update for the floating heart object.
//
// When submerged: fluid drag tracks local flow velocity; buoyancy pushes
// opposite to gravity; local curl drives rotation.
// When airborne: only gravity and a small air damping apply.
// ---------------------------------------------------------------------------
void heart_update(FlipFluid* f, float dt) {
    // Normalize gravity direction from IMU
    float gx_norm = g_imu_x, gy_norm = g_imu_y;
    float glen = sqrtf(gx_norm*gx_norm + gy_norm*gy_norm);
    if (glen < 0.05f) { gx_norm = 0; gy_norm = 1.0f; glen = 1.0f; }
    float gx_n = gx_norm / glen, gy_n = gy_norm / glen;

    // Determine if the heart is currently submerged
    int si = (int)(g_heart.x / SCREEN_W * GRID_W) + 1;
    int sj = (int)(g_heart.y / SCREEN_H * GRID_H) + 1;
    si = si < 1 ? 1 : si > SIM_W-2 ? SIM_W-2 : si;
    sj = sj < 1 ? 1 : sj > SIM_H-2 ? SIM_H-2 : sj;
    float density_here = f->particle_density[si * f->f_num_y + sj];
    float prd = f->particle_rest_density > 0 ? f->particle_rest_density : 1.0f;
    bool in_water = (density_here / prd) > 0.25f;

    if (in_water) {
        float fvx, fvy;
        sample_fluid_vel(f, g_heart.x, g_heart.y, fvx, fvy);

        // Estimate local vorticity (curl) from finite differences for rotation drive
        const float probe = 8.0f;
        float lvx, lvy, rvx, rvy, uvx, uvy, dvx, dvy;
        sample_fluid_vel(f, g_heart.x - probe, g_heart.y, lvx, lvy);
        sample_fluid_vel(f, g_heart.x + probe, g_heart.y, rvx, rvy);
        sample_fluid_vel(f, g_heart.x, g_heart.y - probe, uvx, uvy);
        sample_fluid_vel(f, g_heart.x, g_heart.y + probe, dvx, dvy);
        float curl = ((rvx - lvx) - (uvy - dvy)) / (2.0f * probe);

        // Drag: velocity attracted toward local fluid velocity
        const float drag = 6.0f;
        g_heart.vx += (fvx - g_heart.vx) * drag * dt;
        g_heart.vy += (fvy - g_heart.vy) * drag * dt;

        // Buoyancy: accelerate opposite to gravity direction
        const float buoy = 300.0f;
        g_heart.vx += -gx_n * buoy * dt;
        g_heart.vy += -gy_n * buoy * dt;

        g_heart.vx *= 0.95f;
        g_heart.vy *= 0.95f;

        // Curl drives rotation; heavy damping keeps it subtle
        g_heart.omega += curl * 1.5f;
        g_heart.omega *= 0.65f;

    } else {
        // Airborne: gravity only, negligible air resistance
        const float gravity = GRAVITY_SCALE * 70.0f;
        g_heart.vx += gx_n * gravity * dt;
        g_heart.vy += gy_n * gravity * dt;
        g_heart.vx *= 0.995f;
        g_heart.vy *= 0.995f;
        g_heart.omega *= 0.60f;
    }

    // Integrate position
    g_heart.x += g_heart.vx * dt;
    g_heart.y += g_heart.vy * dt;

    // Soft boundary: bounce with energy loss at screen edges
    const float margin = 14.0f;
    if (g_heart.x < margin)            { g_heart.x = margin;            g_heart.vx *= -0.3f; }
    if (g_heart.x > SCREEN_W - margin) { g_heart.x = SCREEN_W - margin; g_heart.vx *= -0.3f; }
    if (g_heart.y < margin)            { g_heart.y = margin;            g_heart.vy *= -0.3f; }
    if (g_heart.y > SCREEN_H - margin) { g_heart.y = SCREEN_H - margin; g_heart.vy *= -0.3f; }

    g_heart.angle += g_heart.omega * dt;
}

// ---------------------------------------------------------------------------
// draw_heart: render a filled, rotatable heart shape using a parametric curve.
//
// Parametric equations:
//   x = 16 * sin^3(t)
//   y = -(13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t))
//
// 32 sample points are generated, then filled as a triangle fan from the
// center. A highlight ellipse and dark outline are drawn on top.
// ---------------------------------------------------------------------------
void draw_heart(M5Canvas& cv, float cx, float cy, float angle, float size) {
    const int N = 32;
    int16_t pts_x[N], pts_y[N];
    float scale = size / 16.0f;
    float ca = cosf(angle), sa = sinf(angle);

    for (int k = 0; k < N; k++) {
        float t = k * 2.0f * 3.14159f / N;
        float hx =  16.0f * powf(sinf(t), 3.0f);
        float hy = -(13.0f*cosf(t) - 5.0f*cosf(2*t) - 2.0f*cosf(3*t) - cosf(4*t));
        hx *= scale; hy *= scale;
        // Rotate by current angle
        float rx = hx*ca - hy*sa;
        float ry = hx*sa + hy*ca;
        pts_x[k] = (int16_t)(cx + rx);
        pts_y[k] = (int16_t)(cy + ry);
    }

    uint16_t color_fill  = cv.color565(30, 100, 220);
    uint16_t color_light = cv.color565(80, 160, 255);
    uint16_t color_dark  = cv.color565(10,  50, 140);

    // Fill using triangle fan from center
    for (int k = 0; k < N; k++) {
        int nx = (k+1) % N;
        cv.fillTriangle((int16_t)cx, (int16_t)cy,
                        pts_x[k], pts_y[k],
                        pts_x[nx], pts_y[nx],
                        color_fill);
    }

    // Highlight ellipse offset toward upper-left in local space
    float hx_off = -size*0.18f*ca - (-size*0.2f)*sa;
    float hy_off = -size*0.18f*sa + (-size*0.2f)*ca;
    cv.fillEllipse((int16_t)(cx + hx_off), (int16_t)(cy + hy_off),
                   (int16_t)(size*0.22f), (int16_t)(size*0.13f),
                   color_light);

    // Outline
    for (int k = 0; k < N; k++) {
        int nx = (k+1) % N;
        cv.drawLine(pts_x[k], pts_y[k], pts_x[nx], pts_y[nx], color_dark);
    }
}

// ---------------------------------------------------------------------------
// render: draw the fluid density field and HUD to the off-screen canvas,
// then push the sprite to the display.
//
// Each simulation cell maps to a CELL_SIZE x CELL_SIZE pixel block.
// Density is normalized against rest density, gamma-corrected, then mapped
// to a three-stop pink gradient (dark -> mid -> bright).
// ---------------------------------------------------------------------------
void render(FlipFluid* f) {
    canvas.fillSprite(COLOR_BG);

    int n = f->f_num_y;
    float prd = f->particle_rest_density;

    for (int i = 0; i < GRID_W; i++) {
        for (int j = 0; j < GRID_H; j++) {
            // Offset by 1 to skip the solid border ring
            int si = i + 1, sj = j + 1;
            int idx = si * n + sj;

            float d = f->particle_density[idx];
            if (prd > 0) d /= prd;
            float b = d / DENSITY_CLAMP;
            b = b < 0 ? 0 : b > 1 ? 1 : b;

            uint8_t bi = (uint8_t)lrintf(b * 255.0f);
            uint8_t bg = s_gamma_lut[bi];
            float led_val = (float)bg * (LED_VAL_MAX_F / 255.0f);

            if (led_val < 0.5f) continue;  // Skip near-empty air cells

            // Three-stop pink gradient: 0-30% dark pink, 30-65% mid pink, 65-100% bright pink
            float t = led_val / LED_VAL_MAX_F;
            uint8_t r, g, bv;
            if (t < 0.3f) {
                float tt = t / 0.3f;
                r = (uint8_t)(120 * tt);
                g = (uint8_t)(15  * tt);
                bv= (uint8_t)(75  * tt);
            } else if (t < 0.65f) {
                float tt = (t-0.3f)/0.35f;
                r = (uint8_t)(120 + 110*tt);
                g = (uint8_t)(15  +  65*tt);
                bv= (uint8_t)(75  +  75*tt);
            } else {
                float tt = (t-0.65f)/0.35f;
                r = (uint8_t)(230 + 25*tt);
                g = (uint8_t)(80  + 80*tt);
                bv= (uint8_t)(150 + 50*tt);
            }

            int sx = i * CELL_SIZE, sy = j * CELL_SIZE;
            canvas.fillRect(sx, sy, CELL_SIZE, CELL_SIZE,
                canvas.color565(r, g, bv));
        }
    }

    // HUD: CPU bar when monitor is active, otherwise a small status dot
    if (g_pc_on) {
        canvas.setTextColor(0xFFFF);
        canvas.setTextSize(1);
        canvas.setCursor(3, 3);
        canvas.printf("CPU %d%%", (int)g_cpu);
        int bw = (int)(g_cpu/100.0f*55);
        canvas.drawRect(3,13,57,4,0xFFFF);
        uint32_t bc = g_cpu>80?0xF800:g_cpu>50?0xFFE0:0xFBB6;
        canvas.fillRect(4,14,bw,2,bc);
    } else {
        canvas.fillCircle(SCREEN_W-6, 6, 3, canvas.color565(244,143,177));
    }

    if (g_heart_visible) draw_heart(canvas, g_heart.x, g_heart.y, g_heart.angle, 14.0f);

    canvas.pushSprite(0, 0);
}

// ---------------------------------------------------------------------------
// update_imu: read accelerometer and project gravity onto the screen plane.
//
// Physical axis convention for M5Stack StickS3 (chip side up, USB right):
//   ax: positive right  ay: positive up  az: positive out of screen
// Simulation convention: gx positive right, gy positive down.
//
// The mapping is adjusted per rotation so the fluid always responds
// correctly regardless of which way the device is held.
// A first-order low-pass filter (alpha = 0.2) smooths sensor noise.
// ---------------------------------------------------------------------------
void update_imu() {
    float ax, ay, az;
    M5.Imu.getAccel(&ax, &ay, &az);

    int rot = M5.Display.getRotation();
    float screen_gx, screen_gy;
    switch(rot) {
        case 0:  screen_gx = -ay; screen_gy = -ax; break;  // Portrait
        case 1:  screen_gx =  ax; screen_gy = -ay; break;  // Landscape, USB left
        case 2:  screen_gx =  ay; screen_gy =  ax; break;  // Portrait inverted
        case 3:  screen_gx = -ax; screen_gy =  ay; break;  // Landscape, USB right
        default: screen_gx = -ay; screen_gy = -ax; break;
    }

    const float a = 0.2f;
    g_imu_x = g_imu_x*(1-a) + screen_gx*a;
    g_imu_y = g_imu_y*(1-a) + screen_gy*a;

    g_imu_x = g_imu_x < -1 ? -1 : g_imu_x > 1 ? 1 : g_imu_x;
    g_imu_y = g_imu_y < -1 ? -1 : g_imu_y > 1 ? 1 : g_imu_y;
}

// ---------------------------------------------------------------------------
// read_serial: parse CPU usage sent as a float (0-100) over UART.
// If no data is received within PC_TIMEOUT ms, the display reverts to idle.
// ---------------------------------------------------------------------------
void read_serial() {
    while (Serial.available()) {
        String s = Serial.readStringUntil('\n');
        s.trim();
        float val = s.toFloat();
        if (val >= 0 && val <= 100) {
            if (g_cpu_monitor) { g_cpu = val; g_pc_on = true; g_last_rx = millis(); }
        }
    }
    if (g_pc_on && millis()-g_last_rx > PC_TIMEOUT) { g_pc_on=false; g_cpu=0; }
}

// ---------------------------------------------------------------------------
// Setup & Loop
// ---------------------------------------------------------------------------
void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    M5.Display.setRotation(0);
    M5.Display.fillScreen(0);
    M5.Display.setBrightness(150);

    canvas.createSprite(SCREEN_W, SCREEN_H);
    canvas.setColorDepth(16);

    M5.Imu.init();
    Serial.begin(115200);
    gamma_init();

    s_fluid = flip_create(GRID_W, GRID_H, FILL_RATIO);
    if (s_fluid) heart_init(s_fluid);
}

void loop() {
    M5.update();

    static uint32_t last = 0;
    uint32_t now = millis();
    if (now - last < (uint32_t)(1000/SIM_FPS)) return;
    last = now;

    read_serial();
    update_imu();

    // Base gravity from IMU
    float gx = g_imu_x;
    float gy = g_imu_y;

    // When CPU monitor is active, superimpose a sinusoidal perturbation.
    // Frequency scales from 1 Hz at 0% to 4 Hz at 100% CPU.
    // Amplitude is large enough to reverse the vertical gravity component at full load.
    if (g_pc_on && g_cpu > 5.0f) {
        float t_sec = now / 1000.0f;
        float intensity = g_cpu / 100.0f;
        float freq    = 1.0f + intensity * 3.0f;
        float gx_amp  = intensity * GRAVITY_SCALE * 1.0f;
        float gy_amp  = intensity * GRAVITY_SCALE * 2.0f;
        gx += sinf(t_sec * freq * 2.0f * 3.14159f) * gx_amp;
        gy += cosf(t_sec * freq * 2.0f * 3.14159f) * gy_amp;
    }

    // BtnB: toggle floating heart visibility
    if (M5.BtnB.wasPressed()) {
        g_heart_visible = !g_heart_visible;
    }

    // BtnA: toggle CPU monitor mode; clears CPU state when disabled
    if (M5.BtnA.wasPressed()) {
        g_cpu_monitor = !g_cpu_monitor;
        if (!g_cpu_monitor) {
            g_pc_on = false;
            g_cpu   = 0.0f;
        }
    }

    if (s_fluid) {
        flip_step(s_fluid, DT, gx, gy);
        if (g_heart_visible) heart_update(s_fluid, DT);
        render(s_fluid);
    }
}
