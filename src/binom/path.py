import numpy as np
from numpy import sqrt, sin, cos, ceil
from .geometry import intersectBBox, intersectEllipsoid

def pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, plyfile='', outputfile=''):

    kEpsilon = 1e-5

    N = int(ceil(sqrt(nrays)))

    # Ray direction Cartesian unit vector
    dx = sin(ray_zenith) * cos(ray_azimuth)
    dy = sin(ray_zenith) * sin(ray_azimuth)
    dz = cos(ray_zenith)

    path_length = np.zeros(N*N)

    plydata=[]

    if shape == 'polymesh':
        vertices = plydata.elements[0].data
        Nvertices = len(vertices)
        bx_min = 1e6
        bx_max = -1e6
        by_min = 1e6
        by_max = -1e6
        z_min = 1e6
        z_max = -1e6
        for vert in range(0, Nvertices):
            vx = vertices[vert][0] * scale_x
            vy = vertices[vert][1] * scale_y
            vz = vertices[vert][2] * scale_z
            if vx < bx_min:
                bx_min = vx
            if vx > bx_max:
                bx_max = vx
            if vy < by_min:
                by_min = vy
            if vy > by_max:
                by_max = vy
            if vz < z_min:
                z_min = vz
            if vz > z_max:
                z_max = vz
        bbox_sizex = 2*max(abs(bx_max), abs(bx_min)) * (1.0 + kEpsilon)
        bbox_sizey = 2*max(abs(by_max), abs(by_min)) * (1.0 + kEpsilon)
        z_min = z_min
        z_max = z_max * (1.0 + kEpsilon)
    else:
        bbox_sizex = scale_x * (1.0 + kEpsilon)
        bbox_sizey = scale_y * (1.0 + kEpsilon)
        z_min = 0
        z_max = scale_z * (1.0 + kEpsilon)

    sx = bbox_sizex/N
    sy = bbox_sizey/N

    # loop over all rays, which originate at the bottom of the box
    for j in range(0, N):
        for i in range(0, N):

            ox = -0.5*bbox_sizex + (i+0.5)*sx
            oy = -0.5*bbox_sizey + (j+0.5)*sy
            oz = z_min-kEpsilon

            ze = 0
            dr = 0
            while ze <= z_max:

                # Intersect shape
                if  shape == 'ellipsoid':
                    dr = intersectEllipsoid(ox, oy, oz, dx, dy, dz, scale_x, scale_y, scale_z)
                # Intersect bounding box walls
                _, xe, ye, ze = intersectBBox(ox, oy, oz, dx, dy, dz, bbox_sizex, bbox_sizey, 1e6)

                if ze <= z_max:  # intersection below object height -> record path length and periodically cycle ray

                    path_length = np.append(path_length, dr)

                    ox = xe
                    oy = ye
                    oz = ze

                    if abs(ox-0.5*bbox_sizex) < kEpsilon:  # hit +x wall
                        ox = ox - bbox_sizex + kEpsilon
                    elif abs(ox+0.5*bbox_sizex) < kEpsilon:  # hit -x wall
                        ox = ox + bbox_sizex - kEpsilon

                    if abs(oy-0.5*bbox_sizey) < kEpsilon:  # hit +y wall
                        oy = oy - bbox_sizey + kEpsilon
                    elif abs(oy + 0.5 * bbox_sizey) < kEpsilon:  # hit -y wall
                        oy = oy + bbox_sizey - kEpsilon

            path_length[i+j*N] = dr


    if( outputfile != '' ):
        np.savetxt(outputfile, path_length, delimiter=',')

    return path_length[path_length > kEpsilon]

def pathlengthdistribution(
    shape,
    scale_x,
    scale_y,
    scale_z,
    ray_zenith,
    ray_azimuth,
    nrays,
    plyfile="",
    bins=10,
    normalize=True,
):
    pl = pathlengths(shape, scale_x, scale_y, scale_z, ray_zenith, ray_azimuth, nrays, plyfile)
    hist, bin_edges = np.histogram(pl, bins=bins, density=normalize)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {"hist": hist, "bin_centers": bin_centers}