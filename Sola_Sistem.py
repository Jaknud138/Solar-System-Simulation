import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


clock=0
total_simulation_time = 60*60*24*365*83 #s
dt = 3600 #s
clockmax=int(total_simulation_time/dt) #steps

fps = 20
total_animation_time = 360 #s
num_frames = fps * total_animation_time
frame_interval = int(clockmax // num_frames)


#Universal Constants
G=6.6743*(10**-20) #kg, km , s

#Initial motion values - 8/23/2026 12:00 TDB - km, s

x_iMu=-3.340089552401786E+07
y_iMu=3.558922555951048E+07
z_iMu=6.030441948422365E+06
vx_iMu=-4.584765084353999E+01
vy_iMu=-3.088240053086047E+01
vz_iMu=1.681599617803403E+00

x_iV=4.145994433453815E+07
y_iV=-1.013207321289860E+08
z_iV=-3.771361130084455E+06
vx_iV=3.212509723505401E+01
vy_iV=1.329750691996705E+01
vz_iV=-1.670548510366607E+00

x_iE=1.308092587252295E+08
y_iE=-7.638023192361104E+07
z_iE=1.866348695474863E+04
vx_iE=1.441425559186777E+01
vy_iE=2.569458098792079E+01
vz_iE=-1.853883153332347E-03

x_iMa=9.340524173924914E+07
y_iMa=2.045939385448763E+08
z_iMa=2.023086643496931E+06
vx_iMa=-2.111628410012180E+01
vy_iMa=1.211605662165176E+01
vz_iMa=7.716487872520172E-01

x_iJ=-4.883629408385677E+08
y_iJ=6.228634394765074E+08
z_iJ=8.345767942026734E+06
vx_iJ=-1.043868032144486E+01
vy_iJ=-7.446147299280558E+00
vz_iJ=2.645167817211527E-01

x_iSa=1.392371041134147E+09
y_iSa=2.319645957634579E+08
z_iSa=-5.947212886195175E+07
vx_iSa=-2.117932720457131E+00
vy_iSa=9.507193598269062E+00
vz_iSa=-8.101888586045058E-02

x_iU=1.356080785232617E+09
y_iU=2.573139394095057E+09
z_iU=-8.011930029318094E+06
vx_iU=-6.074754765566157E+00
vy_iU=2.857610184266389E+00
vz_iU=8.930679542781905E-02

x_iN=4.464364753351875E+09
y_iN=1.875294988800872E+08
z_iN=-1.067476923503663E+08
vx_iN=-2.638548126536123E-01
vy_iN=5.462717145322250E+00
vz_iN=-1.064150383380253E-01

x_iSu=-2.216118855894471E+05
y_iSu=-7.826359560770710E+05
z_iSu=1.490362408880796E+04
vx_iSu=1.074302738356864E-02
vy_iSu=3.875896543467224E-03
vz_iSu=-2.242517224862839E-04

#Mass values - kg

m_1=1.9885*(10**30)
m_2=3.3011*(10**23)
m_3=4.86732*(10**24)
m_4=6.0457*(10**24)
m_5=6.4171*(10**23)
m_6=1.8982*(10**27)
m_7=5.683*(10**26)
m_8=8.681*(10**25)
m_9=1.0241*(10**26)

m=np.array([m_1,m_2,m_3,m_4,m_5,m_6,m_7,m_8,m_9])

pos=np.array([[x_iSu,y_iSu,z_iSu],
              [x_iMu,y_iMu,z_iMu],
              [x_iV,y_iV,z_iV],
              [x_iE,y_iE,z_iE],
              [x_iMa,y_iMa,z_iMa],
              [x_iJ,y_iJ,z_iJ],
              [x_iSa,y_iSa,z_iSa],
              [x_iU,y_iU,z_iU],
              [x_iN,y_iN,z_iN]])

vel = np.array([
    [vx_iSu, vy_iSu, vz_iSu],
    [vx_iMu, vy_iMu, vz_iMu],
    [vx_iV, vy_iV, vz_iV],
    [vx_iE, vy_iE, vz_iE],
    [vx_iMa, vy_iMa, vz_iMa],
    [vx_iJ, vy_iJ, vz_iJ],
    [vx_iSa, vy_iSa, vz_iSa],
    [vx_iU, vy_iU, vz_iU],
    [vx_iN, vy_iN, vz_iN],
])

x_data = np.zeros((9, num_frames+1))
y_data = np.zeros((9, num_frames+1))
z_data = np.zeros((9, num_frames+1))


for clock in range (clockmax):
    for i in range (9):
        for j in range (9):
            if i!=j:
                r_ij=np.linalg.norm(pos[i]-pos[j])
                vel[i]+=dt*G*m[j]*(pos[j]-pos[i])/(r_ij**3)    

    for i in range (9):
        pos[i]+=dt*vel[i]

    if clock % frame_interval == 0:
        frame_index = min(clock // frame_interval, num_frames)
        x_data[:, frame_index] = pos[:, 0]
        y_data[:, frame_index] = pos[:, 1]
        z_data[:, frame_index] = pos[:, 2]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

line_Mu,=ax.plot([], [], [], lw =2, color='grey')
line_V,=ax.plot([], [], [], lw =2, color='darkred')
line_E,=ax.plot([], [], [], lw =2, color='blue')
line_Ma,=ax.plot([], [], [], lw =2, color='orange')
line_J,=ax.plot([], [], [], lw =2, color='tan')
line_Sa,=ax.plot([], [], [], lw =2, color='lightgreen')
line_U,=ax.plot([], [], [], lw =2, color='lightblue')
line_N,=ax.plot([], [], [], lw =2, color='darkblue')

point_Su,=ax.plot([], [], [], 'o', color='yellow', label='Sun')
point_Mu,=ax.plot([], [], [], 'o', color='grey', label='Mercury')
point_V,=ax.plot([], [], [], 'o', color='darkred', label='Venus')
point_E,=ax.plot([], [], [], 'o', color='blue', label='Earth')
point_Ma,=ax.plot([], [], [], 'o', color='orange', label='Mars')
point_J,=ax.plot([], [], [], 'o', color='tan', label='Jupiter')
point_Sa,=ax.plot([], [], [], 'o', color='lightgreen', label='Saturn')
point_U,=ax.plot([], [], [], 'o', color='lightblue', label='Uranus')
point_N,=ax.plot([], [], [], 'o', color='darkblue', label='Neptune')

#currently set for seeing the closer planets, change order of magnitude by 1 for x and y and 2 for z to see outer planets
ax.set_xlim(-5e8, 5e8)
ax.set_ylim(-5e8, 5e8)
ax.set_zlim(-5e8, 5e8)

ax.set_xlabel('X Position (km)')
ax.set_ylabel('Y Position (km)')
ax.set_zlabel('Z Position (km)')

ax.set_title('Sola Sistem')

ax.legend()

def init():
    line_Mu.set_data([], [])
    line_Mu.set_3d_properties([])
    line_V.set_data([], [])
    line_V.set_3d_properties([])
    line_E.set_data([], [])
    line_E.set_3d_properties([])
    line_Ma.set_data([], [])
    line_Ma.set_3d_properties([])
    line_J.set_data([], [])
    line_J.set_3d_properties([])
    line_Sa.set_data([], [])
    line_Sa.set_3d_properties([])
    line_U.set_data([], [])
    line_U.set_3d_properties([])
    line_N.set_data([], [])
    line_N.set_3d_properties([])

    point_Su.set_data([], [])
    point_Su.set_3d_properties([])
    point_Mu.set_data([], [])
    point_Mu.set_3d_properties([])
    point_V.set_data([], [])
    point_V.set_3d_properties([])
    point_E.set_data([], [])
    point_E.set_3d_properties([])
    point_Ma.set_data([], [])
    point_Ma.set_3d_properties([])
    point_J.set_data([], [])
    point_J.set_3d_properties([])
    point_Sa.set_data([], [])
    point_Sa.set_3d_properties([]) 
    point_U.set_data([], [])
    point_U.set_3d_properties([])
    point_N.set_data([], [])
    point_N.set_3d_properties([])
    return line_Mu, line_V, line_E, line_Ma, line_J, line_Sa, line_U, line_N, point_Su, point_Mu, point_V, point_E, point_Ma, point_J, point_Sa, point_U, point_N

orbital_periods = np.zeros(9)
tail_lengths = np.zeros(9, dtype=int)

for planet_tag in range(1, 9):
    axis = np.mean([(x_data[planet_tag])**2+(y_data[planet_tag])**2+(z_data[planet_tag])**2])
    orbital_periods[planet_tag] = 2*np.pi*np.sqrt(axis**3/(G*m_1))
    tail_lengths[planet_tag] = int(0.7*int(np.round(num_frames * float((orbital_periods[planet_tag] / total_simulation_time))))) #multiplied by 0.7 to (hopefully)dsplay only 70% of the orbits as tails

def update(frame):
    line_Mu.set_data(x_data[1, max(0, (frame - tail_lengths[1])):frame], y_data[1, max(0, (frame - tail_lengths[1])):frame])
    line_Mu.set_3d_properties(z_data[1, max(0,(frame - tail_lengths[1])):frame])
    line_V.set_data(x_data[2, max(0, (frame - tail_lengths[2])):frame], y_data[2, max(0, (frame - tail_lengths[2])):frame])
    line_V.set_3d_properties(z_data[2, max(0,(frame - tail_lengths[2])):frame])
    line_E.set_data(x_data[3, max(0, (frame - tail_lengths[3])):frame], y_data[3, max(0, (frame - tail_lengths[3])):frame])
    line_E.set_3d_properties(z_data[3, max(0,(frame - tail_lengths[3])):frame])
    line_Ma.set_data(x_data[4, max(0, (frame - tail_lengths[4])):frame], y_data[4, max(0, (frame - tail_lengths[4])):frame])
    line_Ma.set_3d_properties(z_data[4, max(0,(frame - tail_lengths[4])):frame])
    line_J.set_data(x_data[5, max(0, (frame - tail_lengths[5])):frame], y_data[5, max(0, (frame - tail_lengths[5])):frame])
    line_J.set_3d_properties(z_data[5, max(0,(frame - tail_lengths[5])):frame])
    line_Sa.set_data(x_data[6, max(0, (frame - tail_lengths[6])):frame], y_data[6, max(0, (frame - tail_lengths[6])):frame])
    line_Sa.set_3d_properties(z_data[6, max(0,(frame - tail_lengths[6])):frame])
    line_U.set_data(x_data[7, max(0, (frame - tail_lengths[7])):frame], y_data[7, max(0, (frame - tail_lengths[7])):frame])
    line_U.set_3d_properties(z_data[7, max(0,(frame - tail_lengths[7])):frame])
    line_N.set_data(x_data[8, max(0, (frame - tail_lengths[8])):frame], y_data[8, max(0, (frame - tail_lengths[8])):frame])
    line_N.set_3d_properties(z_data[8, max(0,(frame - tail_lengths[8])):frame])

    point_Su.set_data(x_data[0, frame:frame + 1], y_data[0, frame:frame + 1])
    point_Su.set_3d_properties(z_data[0, frame:frame + 1])
    point_Mu.set_data(x_data[1, frame:frame + 1], y_data[1, frame:frame + 1])
    point_Mu.set_3d_properties(z_data[1, frame:frame + 1])
    point_V.set_data(x_data[2, frame:frame + 1], y_data[2, frame:frame + 1])
    point_V.set_3d_properties(z_data[2, frame:frame + 1])
    point_E.set_data(x_data[3, frame:frame + 1], y_data[3, frame:frame + 1])
    point_E.set_3d_properties(z_data[3, frame:frame + 1])
    point_Ma.set_data(x_data[4, frame:frame + 1], y_data[4, frame:frame + 1])
    point_Ma.set_3d_properties(z_data[4, frame:frame + 1])
    point_J.set_data(x_data[5, frame:frame + 1], y_data[5, frame:frame + 1])
    point_J.set_3d_properties(z_data[5, frame:frame + 1])
    point_Sa.set_data(x_data[6, frame:frame + 1], y_data[6, frame:frame + 1])
    point_Sa.set_3d_properties(z_data[6, frame:frame + 1])
    point_U.set_data(x_data[7, frame:frame + 1], y_data[7, frame:frame + 1])
    point_U.set_3d_properties(z_data[7, frame:frame + 1])
    point_N.set_data(x_data[8, frame:frame + 1], y_data[8, frame:frame + 1])
    point_N.set_3d_properties(z_data[8, frame:frame + 1])
    return line_Mu, line_V, line_E, line_Ma, line_J, line_Sa, line_U, line_N, point_Su, point_Mu, point_V, point_E, point_Ma, point_J, point_Sa, point_U, point_N

ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=50, blit=True)

if animation.writers.is_available("ffmpeg"):
    ani.save("3d_simulation.mp4", writer="ffmpeg", fps=fps, dpi=150)
else:
    ani.save("3d_simulation1.gif", writer=animation.PillowWriter(fps=fps), dpi=150)

plt.show()