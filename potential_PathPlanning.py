from collections import deque
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot.xlabel
# import matplotlib.pyplot.ylabel

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 0.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 10

show_animation = True


def calc_potential_field(goal_x, goal_y, obstacle_x, obstacle_y, reso, rr, start_x, start_y):
    minx = min(min(obstacle_x), start_x, goal_x) - AREA_WIDTH / 2.0
    miny = min(min(obstacle_y), start_y, goal_y) - AREA_WIDTH / 2.0
    maxx = max(max(obstacle_x), start_x, goal_x) + AREA_WIDTH / 2.0
    maxy = max(max(obstacle_y), start_y, goal_y) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, goal_x, goal_y)
            uo = calc_repulsive_potential(x, y, obstacle_x, obstacle_y, rr,goal_x, goal_y)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, goal_x, goal_y):
    return 0.5 * KP * np.hypot(x - goal_x, y - goal_y)


def calc_repulsive_potential(x, y, obstacle_x, obstacle_y, rr,goal_x, goal_y):
    # search nearest obstacle
    minid = -1
    dz = np.hypot(x - goal_x, y - goal_y)
    dmin = float("inf")
    for i, _ in enumerate(obstacle_x):
        d = np.hypot(x - obstacle_x[i], y - obstacle_y[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - obstacle_x[minid], y - obstacle_y[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return (0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2) + 2 / dz
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    # all the 8 neighbouring cells to be checked
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(start_x, start_y, goal_x, goal_y, obstacle_x, obstacle_y, reso, rr):
    # calc potential field
    pmap, minx, miny = calc_potential_field(goal_x, goal_y, obstacle_x, obstacle_y, reso, rr, start_x, start_y)

    # search path
    d = np.hypot(start_x - goal_x, start_y - goal_y)
    ix = round((start_x - minx) / reso)
    iy = round((start_y - miny) / reso)
    gix = round((goal_x - minx) / reso)
    giy = round((goal_y - miny) / reso)
    print(gix,giy)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "oy")
        plt.plot(gix, giy, "ok")
        plt.annotate("GOAL", xy=(gix + 2, giy + 2))
        # plt.annotate("GOAL", xy=(goal_y, goal_y))
        plt.annotate("START", xy=(start_x, start_y), color='yellow')
        plt.axis(True)

    path_x, path_y = [start_x], [start_y]
    motion = get_motion_model()
    previous_ids = deque()

    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(goal_x - xp, goal_y - yp)
        path_x.append(xp)
        path_y.append(yp)

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.12)

    print("Goal!!")

    return path_x, path_y


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)
    
    


def main():
    print("potential_field_planning start")

    start_x = 0.0  # start x position [m]
    start_y = 0.0  # start y position [m]
    goal_x = 16.0  # goal x position [m]
    goal_y = 38.0  # goal y position [m]
    grid_size = 0.5 # potential grid size [m]
    robot_radius = 5.0 # robot radius [m]

    ox = [10.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 25.0, 28.0, 20.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,15.0,18.0,18.0,18.0,30.0,30.0,30.0]  # obstacle x position list [m]
    oy = [10.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 26.0, 25.0, 28.0, 30.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20, 21,22,23,7.0,8.0,9.0,45.0,46.0,47.0] 

    # ox, oy = [], []
    # for i in range(-10, 60):
    #     ox.append(i)
    #     oy.append(-10.0)
        
    # for i in range(-10, 60):
    #     ox.append(i)
    #     oy.append(-10.0)
        
    # for i in range(-10, 60):
    #     ox.append(60.0)
    #     oy.append(i)
        
    # for i in range(-10, 61):
    #     ox.append(i)
    #     oy.append(60.0)
        
    # for i in range(-10, 61):
    #     ox.append(-10.0)
    #     oy.append(i)
    
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
        
    # for i in range(-10, 40):
    #     ox.append(21.0)
    #     oy.append(i)
    
    # for i in range(-10, 40):
    #     ox.append(22.0)
    #     oy.append(i)
    
    # for i in range(-10, 40):
    #     ox.append(23.0)
    #     oy.append(i)
        
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)
        
   

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X Axis (m)")
        plt.ylabel("Y Axis (m)")

    # path generation
    _, _ = potential_field_planning(
        start_x, start_y, goal_x, goal_y, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
