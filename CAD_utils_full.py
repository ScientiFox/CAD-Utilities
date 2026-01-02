###
#
# A selection of programmatic tools for manipulating CAD mesh files- mainly intended for STLs, but
#   also supporting most anything that can be converted into a 9D mesh file
#
###

#General imports
import math,time,random

#File handling
import glob,shutil,copy

#Numperical processing
import numpy as np
import cv2
from scipy import ndimage
from scipy.special import comb

#CAD packages
from stl import mesh
import pyvista as pv

###########################
# Processing Functions
###########################


def subsample_mesh(_mesh,N = 1000):
    # Simple function to subsample a mesh down - diagnostic and display
    #   Selects a sub-proportions of points randomly to represent the shape

    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    centers = np.zeros(np.shape(mesh_pt1_rot))
    centers[:,0] = (mesh_pt1_rot[:,0]+mesh_pt2_rot[:,0]+mesh_pt3_rot[:,0])/3
    centers[:,1] = (mesh_pt1_rot[:,1]+mesh_pt2_rot[:,1]+mesh_pt3_rot[:,1])/3
    centers[:,2] = (mesh_pt1_rot[:,2]+mesh_pt2_rot[:,2]+mesh_pt3_rot[:,2])/3

    ops = []
    indicies = []

    #Pull N-many points from the mesh as representative
    for a in range(N):
        r = random.randint(0,len(centers)-1)
        ops = ops + [centers[r,:]]
        indicies = indicies + [r]

    return ops,indicies

def downsample(fil,D_SAMP=120000,lead_string="reduced ",hole_size=1000,fix_holes=True,decimate_mode=0):
    #Function to downsample STL file to D_SAMP triangles

    #Open STL in VTK format
    p_mesh = pv.PolyData(fil)

    #Collect the number of faces
    p_nf_init = p_mesh.n_cells

    #Calculate downscale proportion- proportion reduced by, not number of faces to
    prop = 1.0-(1.0*D_SAMP)/p_mesh.n_faces

    #Selecting decimation alg to use
    if decimate_mode == 0:
        #decimate_pro- default because it's considered industry standard
        #  preserve_topology prevents use of risky heuristics
        p_mesh = p_mesh.decimate_pro(prop,preserve_topology=True)

    if decimate_mode == 1:
        #Standard decimate- slower, ostensibly more stable
        p_mesh = p_mesh.decimate(prop)

    if decimate_mode == 2:
        #Boundary based decimation, ostensibly faster, but more risky
        p_mesh = p_mesh.decimate_boundary(target_reduction=prop)

    if fix_holes:
        #Optional hole-repair step, segfault risk from package noted
        p_mesh = p_mesh.fill_holes(hole_size)

    #check final number of faces
    p_nf_fin = p_mesh.n_faces

    #Save transformed mesh
    p_mesh.save(lead_string+fil.split("\\")[-1])
    #print(lead_string+fil.split("\\")[-1])

    #Return numbers of faces and final file name
    return p_mesh,p_nf_init,p_nf_fin,lead_string+fil.split("\\")[-1]


####
#AutoAlign
####

def auto_center(fil,lead_string="centered "):
    # Function to automatically pose the STL in standard layout
    #   finds the COM of the object, moves the lowest point of the
    #   z-axis to 0 height, and poses XY over 0,0

    #Load the STL in VTK format
    _mesh = mesh.Mesh.from_file(fil)

    #Get the corners of the triangles
    mesh_pt1 = _mesh[:,:3]
    mesh_pt2 = _mesh[:,3:6]
    mesh_pt3 = _mesh[:,6:9]

    #Calculate the center points of the triangles
    centers = np.zeros(np.shape(mesh_pt1))
    centers[:,0] = (mesh_pt1[:,0]+mesh_pt2[:,0]+mesh_pt3[:,0])/3
    centers[:,1] = (mesh_pt1[:,1]+mesh_pt2[:,1]+mesh_pt3[:,1])/3
    centers[:,2] = (mesh_pt1[:,2]+mesh_pt2[:,2]+mesh_pt3[:,2])/3

    #Calculate the center of mass of the model
    COM = np.sum(centers,axis=0)/np.shape(centers)[0]

    #Get the minimum height of the model- least Z of all triangle corners
    min_H = min([min(mesh_pt1[:,2]),min(mesh_pt2[:,2]),min(mesh_pt3[:,2])])

    #Move the model XY coords to put the COM over (0,0,-)
    _mesh[:,:2] = mesh_pt1[:,:2] - COM[:2]
    _mesh[:,3:5] = mesh_pt2[:,:2] - COM[:2]
    _mesh[:,6:8] = mesh_pt3[:,:2] - COM[:2]

    #Lower the model to have the minimum Z at (-,-,0)
    _mesh[:,2] = mesh_pt1[:,2] - min_H
    _mesh[:,5] = mesh_pt2[:,2] - min_H
    _mesh[:,8] = mesh_pt3[:,2] - min_H

    #Save the transformed model
    _mesh.save(lead_string+fil.split("\\")[-1])

    #Return the XY coords of the COM, least height point, and new file name
    return(COM[:2],min_H,lead_string+fil.split("\\")[-1]) 

def auto_rotate_full(fil,lead_string="rotated ",do_XY=True,do_Z=True,do_COR=True):
    # Function to automatically orient the model along a standard axis
    #   Identifies a standardized orientation vector from the vectors from mesh patch
    #   centers to the center of mass, and selects the 50% that are closest to the COM
    #   then transforms the object by rotation to align the Z component of that vector
    #   with the reference frame, then the XY components to <1,0,0> and <0,1,0>,
    #   followed by a final small angle correction to the Z axis to account for
    #   errors based on the whole body's orientation

    #Load the STL in VTK format
    _mesh = mesh.Mesh.from_file(fil)

    #Get the triangle corner points
    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    #Calculate the traingle centers
    centers = np.zeros(np.shape(mesh_pt1_rot))
    centers[:,0] = (mesh_pt1_rot[:,0]+mesh_pt2_rot[:,0]+mesh_pt3_rot[:,0])/3
    centers[:,1] = (mesh_pt1_rot[:,1]+mesh_pt2_rot[:,1]+mesh_pt3_rot[:,1])/3
    centers[:,2] = (mesh_pt1_rot[:,2]+mesh_pt2_rot[:,2]+mesh_pt3_rot[:,2])/3

    #Calculate the center of mass of the object
    COM = np.sum(centers,axis=0)/np.shape(centers)[0]

    #Get the vectors of each point relative to the COM
    rel_vecs = centers - COM

    #Calculate the size of the relative vectors
    rel_mags = np.sqrt(np.sum(rel_vecs**2,axis=1))

    #Get the average length of the relative vectors
    rel_mag_avg = np.average(rel_mags,axis=0)

    #Calculate an orientation vector set
    orient = rel_vecs*1.0

    #Select from orientation vector only the points at the 50% least distance to the COM
    orient[:,0] = orient[:,0]*(rel_mags < rel_mag_avg)
    orient[:,1] = orient[:,1]*(rel_mags < rel_mag_avg)
    orient[:,2] = orient[:,2]*(rel_mags < rel_mag_avg)

    #Get the average vector of the orientation vectors
    orient = np.sum(orient,axis=0)
    orient = orient/np.sum((rel_mags < rel_mag_avg)*1.0)

    #Normalize the orientation vector
    mag = math.sqrt(np.sum(orient**2))
    orient = orient/mag

    #Get the projection of the orientation vector on the XY plane
    proj_xy = np.array([[orient[0],orient[1],0]])/math.sqrt(orient[0]**2 + orient[1]**2)

    #Calculate the XY-plane rotation of the orientation vector
    pxy_dot = np.dot(np.array([[0,1,0]]),proj_xy.T)
    theta_XY = math.acos(pxy_dot)

    #Calculate the Z-axis rotation of the orientation vector
    theta_Z = math.pi/2 - math.acos(orient[2])

    if (do_Z):
        #If performing the bulk Z-axis rotation

        #Calculate the rotation transform 
        ct = math.cos(theta_Z)
        st = math.sin(theta_Z)
        rot_arr = np.array([[1,0,0],[0,ct,-1*st],[0,st,ct]])

        #Apply rotation transform to the shifted triangle points
        mesh_pt1_rot = np.dot(mesh_pt1_rot-COM,rot_arr)+COM
        mesh_pt2_rot = np.dot(mesh_pt2_rot-COM,rot_arr)+COM
        mesh_pt3_rot = np.dot(mesh_pt3_rot-COM,rot_arr)+COM

    #XY rotation
    if (do_XY):
        #If performing the bulk XY plane rotation

        #Calculate the rotation transform
        ct = math.cos(theta_XY)
        st = math.sin(theta_XY)
        rot_arr = np.array([[ct,-1*st,0],[st,ct,0],[0,0,1]])

        #Rotate the triangle points
        mesh_pt1_rot = np.dot(mesh_pt1_rot-COM,rot_arr)+COM
        mesh_pt2_rot = np.dot(mesh_pt2_rot-COM,rot_arr)+COM
        mesh_pt3_rot = np.dot(mesh_pt3_rot-COM,rot_arr)+COM

    if (do_COR):
        #If performing the small angle Z correction

        #Calculate the new centers (including prior rotation)
        centers = np.zeros(np.shape(mesh_pt1_rot))
        centers[:,0] = (mesh_pt1_rot[:,0]+mesh_pt2_rot[:,0]+mesh_pt3_rot[:,0])/3
        centers[:,1] = (mesh_pt1_rot[:,1]+mesh_pt2_rot[:,1]+mesh_pt3_rot[:,1])/3
        centers[:,2] = (mesh_pt1_rot[:,2]+mesh_pt2_rot[:,2]+mesh_pt3_rot[:,2])/3

        #Caculate new COM (should be same, but just in case)
        COM = np.sum(centers,axis=0)/np.shape(centers)[0]

        #Calculate new relative vectors
        rel_vecs = centers - COM

        #Calculate theta orientation of the full body
        t_body = -1*np.abs(np.average(0.5*(np.arctan(rel_vecs[:,1]/rel_vecs[:,2]))))

        #Calculate the transform matrix
        ct = math.cos(t_body)
        st = math.sin(t_body)
        rot_arr = np.array([[1,0,0],[0,ct,-1*st],[0,st,ct]])

        #Rotate trinagle points
        mesh_pt1_rot = np.dot(mesh_pt1_rot-COM,rot_arr)+COM
        mesh_pt2_rot = np.dot(mesh_pt2_rot-COM,rot_arr)+COM
        mesh_pt3_rot = np.dot(mesh_pt3_rot-COM,rot_arr)+COM

    #Set original mesh to new rotated points
    _mesh[:,:3] = mesh_pt1_rot
    _mesh[:,3:6] = mesh_pt2_rot
    _mesh[:,6:9] = mesh_pt3_rot

    #Save the transformed mesh
    _mesh.save(lead_string+fil.split("\\")[-1])

    #Diagnostic print function
    #print(round(180*theta_Z/math.pi,3),round(180*theta_XY/math.pi,3),round(180*t_body/math.pi,3))

    #Output bulk Z angle, final file name, bulk XY angle, and fine adjust angle
    return theta_Z,lead_string+fil.split("\\")[-1],theta_XY,t_body

def get_principal_axis2(_mesh,_name="",_mode="BOT"):
    # The flagship function, which uses a sequence of gerometric descriptors to identify
    #   reliable primary axes within a model based on approximants of fully orderable
    #   metrics and uses those to construct a consistent orientation
    #   Begins by picking a random triangle embedded within the body, which is then 
    #   iterated a number of times with new random points swapped in for any of the three
    #   prior points. When a larger triangle is found, the new point is added in. This
    #   converges towards the largest embedded triangle which can fit within the body
    #   Next, we get the normal to the triangle, and project the mesh points onto the
    #   triangle plane, and grab all points within a nearby cylinder of the triangle
    #   points
    #   We find the lowest and highest points in these cylinders to define a triangle
    #   sandwich which lays over and under the body, defining a prism affixed to the
    #   object. 
    #   We then define a 'top' and 'bottom' for the object by comparing the total area of
    #   the mesh near the two triangles, assigning 'down' to be towards the side with
    #   greater area
    #   Then, we grab one of the base triangle points and calculate a vector perpendicular
    #   to it and the normal line, to make a proper axis, followed by rotating that vector
    #   in-plane to identify a plane slace that most closely divides the model in half
    #   (using an iterative check instead of direct calculation because of possible
    #   patch errors) to define a second fixed axis to go with our normal
    #   Finally, we construct the third axis to be perpendicular to the prior two, and
    #   and calculate all the transforms to align the new axes with the reference frame
    #   and translate the body COM to 0,0,0

    #get mesh points
    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    #get representative triangle points
    centers = mesh_pt1_rot

    #Calculate patch areas
    v1 = mesh_pt1_rot - mesh_pt2_rot
    v2 = mesh_pt3_rot - mesh_pt2_rot
    vp = np.cross(v1,v2)
    vp_mag = np.sqrt(np.sum(vp**2,axis=1))

    #Get weighted centers
    cen_a = np.copy(centers)
    cen_a[:,0] = cen_a[:,0]*vp_mag
    cen_a[:,1] = cen_a[:,1]*vp_mag
    cen_a[:,2] = cen_a[:,2]*vp_mag
    COM = np.sum(cen_a,axis=0)/np.sum(vp_mag)

    #Count number of points
    N_pts = np.shape(centers)[0]

    #Grab 3 random points for an initial triangle
    n_1 = random.randint(0,N_pts-1)
    n_2 = random.randint(0,N_pts-1)
    n_3 = random.randint(0,N_pts-1)
    pt_1 = centers[n_1,:]
    pt_2 = centers[n_2,:]
    pt_3 = centers[n_3,:]

    #Initialize the area metric
    met_A = np.cross(pt_1-pt_2,pt_3-pt_2)
    met_A = np.sqrt(np.sum(met_A**2))

    pts_tri = []

    #Do iterations of random samples
    for i in range(SAMPLE_ITERATIONS):
        #Get a new point
        n_n = random.randint(0,N_pts-1)
        pt_n = centers[n_n,:]

        #Check the areas when swapping in new point to the triangle at each former point
        met_A1 = np.cross(pt_n-pt_2,pt_3-pt_2)
        met_A1 = np.sqrt(np.sum(met_A1**2))

        met_A2 = np.cross(pt_1-pt_n,pt_3-pt_n)
        met_A2 = np.sqrt(np.sum(met_A2**2))

        met_A3 = np.cross(pt_1-pt_2,pt_n-pt_2)
        met_A3 = np.sqrt(np.sum(met_A3**2))

        #List off all four triangle versions sorted by area
        mets = [(met_A,0),(met_A1,1),(met_A2,2),(met_A3,3)]
        mets.sort(key=lambda x:x[0],reverse=True)

        #Replace a point with the new one if the area is most increased by the substitution 
        met_sel = mets[0]
        if met_sel[1] == 0:
            pass
        elif met_sel[1] == 1:
            met_A = met_A1
            pt_1 = pt_n
        elif met_sel[1] == 2:
            met_A = met_A2
            pt_2 = pt_n
        elif met_sel[1] == 3:
            met_A = met_A3
            pt_3 = pt_n
            
        pts_tri = pts_tri + [pt_1,pt_2,pt_3]

    #Save the triangle points
    save_points_bare(np.array(pts_tri),"triangles")

    init_tri_pts = [pt_1,pt_2,pt_3]

    #Construct a vector perpendicular to the large triangle
    v_perp = np.cross(pt_1-pt_2,pt_3-pt_2)
    v_perp = v_perp/np.sqrt(np.sum(v_perp*v_perp))

    pts_tp = []
    pts_tp = pts_tp + [pt_1+i*v_perp*0.2 for i in range(50)]
    save_points_bare(np.array(pts_tp),"perp_tri")

    #Get projections of the centers onto the triangle plane
    centers_perp = (centers-COM) - np.dot(np.array([np.dot(centers-COM,v_perp)]).T,np.ones((1,3)))*v_perp + COM

    #Get the projections of the triangle corner distances onto the plane normal
    pt_1_perp = (pt_1-COM) - np.dot(pt_1-COM,v_perp)*v_perp+COM
    pt_2_perp = (pt_2-COM) - np.dot(pt_2-COM,v_perp)*v_perp+COM
    pt_3_perp = (pt_3-COM) - np.dot(pt_3-COM,v_perp)*v_perp+COM

    #Grab the set of points within planar distance thresh of the triangle corners
    pts_pt_1 = 1*(np.sqrt(np.sum((centers_perp - pt_1_perp)**2,axis=1))<PLANAR_DIST_THRESH)
    pts_pt_2 = 1*(np.sqrt(np.sum((centers_perp - pt_2_perp)**2,axis=1))<PLANAR_DIST_THRESH)
    pts_pt_3 = 1*(np.sqrt(np.sum((centers_perp - pt_3_perp)**2,axis=1))<PLANAR_DIST_THRESH)

    #Save the points in the cylinder near the triangle points
    save_points_bare(centers[pts_pt_1==1],"tri_near")

    #Get the projections of the relative vectors onto the triangle normal
    proj_perp = np.dot((centers-COM),v_perp)

    #Select out the distances along the normal to the selected points near the corners
    proj_perp_1 = pts_pt_1*proj_perp
    proj_perp_2 = pts_pt_2*proj_perp
    proj_perp_3 = pts_pt_3*proj_perp

    #Pick the furthest point along the normal from the filters
    proj_perp_1M = 1*(proj_perp_1 == np.max(proj_perp_1[pts_pt_1==1]))
    proj_perp_2M = 1*(proj_perp_2 == np.max(proj_perp_2[pts_pt_2==1]))
    proj_perp_3M = 1*(proj_perp_3 == np.max(proj_perp_3[pts_pt_3==1]))

    #Pick the lowest point along the normal from the filters
    proj_perp_1m = 1*(proj_perp_1 == np.min(proj_perp_1[pts_pt_1==1]))
    proj_perp_2m = 1*(proj_perp_2 == np.min(proj_perp_2[pts_pt_2==1]))
    proj_perp_3m = 1*(proj_perp_3 == np.min(proj_perp_3[pts_pt_3==1]))

    #Select out the actual points using the filters for the maxima
    top_pt_1 = np.sum(centers*np.dot(np.array([proj_perp_1M]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_1M)
    top_pt_2 = np.sum(centers*np.dot(np.array([proj_perp_2M]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_2M)
    top_pt_3 = np.sum(centers*np.dot(np.array([proj_perp_3M]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_3M)

    #Select out the actual points using the filters for the minima
    bot_pt_1 = np.sum(centers*np.dot(np.array([proj_perp_1m]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_1m)
    bot_pt_2 = np.sum(centers*np.dot(np.array([proj_perp_2m]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_2m)
    bot_pt_3 = np.sum(centers*np.dot(np.array([proj_perp_3m]).T,np.ones((1,3))),axis=0)/np.sum(proj_perp_3m)

    #Grab and normalize the plane normals from the top and bottom triangles
    top_norm = np.cross(top_pt_1-top_pt_2,top_pt_3-top_pt_2)
    top_norm = top_norm/np.sqrt(np.sum(top_norm**2))
    bot_norm = np.cross(bot_pt_1-bot_pt_2,bot_pt_3-bot_pt_2)
    bot_norm = bot_norm/np.sqrt(np.sum(bot_norm**2))

    #Calculate triangle centers
    top_cen = (top_pt_1+top_pt_2+top_pt_3)/3
    bot_cen = (bot_pt_1+bot_pt_2+bot_pt_3)/3

    #Save dotted lines for the two triangle normals
    top_line = [top_cen + 0.2*i*top_norm for i in range(50)]
    bot_line = [bot_cen - 0.2*i*bot_norm for i in range(50)]
    save_points_bare(np.array(top_line+bot_line),"tri_norms")

    #Get the relative vectors to the model from the top and bottom triangle centers
    top_vecs = centers - top_cen
    bot_vecs = centers - bot_cen

    #Project the triangle-relative vectors onto the respective normals
    top_vecs_proj = np.dot(top_vecs,top_norm)
    bot_vecs_proj = np.dot(bot_vecs,bot_norm)

    #Select out a threshold range distant to the triangles in space
    top_vecs_proj_th = 1*(np.abs(top_vecs_proj)<TRIANGLE_DIST_THRESH)
    bot_vecs_proj_th = 1*(np.abs(bot_vecs_proj)<TRIANGLE_DIST_THRESH)

    #Calculate the total area on the model occupied by in-threshold points from above
    top_area = np.sum(top_vecs_proj_th*vp_mag)
    bot_area = np.sum(bot_vecs_proj_th*vp_mag)

    #Select the actual top from bottom by picking the 'base' to be the triangle which covers the greater area
    if (top_area > bot_area):
        #Set the base values to those of the triangle with the greater coverage
        base_pts = centers*np.dot(np.array([top_vecs_proj_th]).T,np.ones((1,3)))

        base_cen = top_cen
        upper_cen = bot_cen

        base_coords = [top_pt_1,top_pt_2,top_pt_3]
        upper_coords = [bot_pt_1,bot_pt_2,bot_pt_3]

        base_norm = top_norm
        upper_norm = bot_norm
        
        #Pick up from down normal based on base normal needing to aim towards the other triangle
        v_up = bot_cen-top_cen
        if np.dot(base_norm,v_up) < 0.0:
            base_norm = -1*base_norm
        if np.dot(upper_norm,v_up) < 0.0:
            upper_norm = -1*upper_norm
    else:
        #Same as other clause except for using the flip triangles
        base_pts = centers*np.dot(np.array([bot_vecs_proj_th]).T,np.ones((1,3)))

        base_cen = bot_cen
        upper_cen = top_cen

        base_coords = [bot_pt_1,bot_pt_2,bot_pt_3]
        upper_coords = [top_pt_1,top_pt_2,top_pt_3]

        base_norm = bot_norm
        upper_norm = top_norm

        v_up = top_cen-bot_cen
        if np.dot(base_norm,v_up) < 0.0:
            base_norm = -1*base_norm
        if np.dot(upper_norm,v_up) < 0.0:
            upper_norm = -1*upper_norm

    #Grab the vector from the COM to the center of the base triangle for offset height
    v_base_COM = COM-base_cen
    v_bC_proj = np.dot(v_base_COM,base_norm)

    #Grab a vector in the base plane, normalize, and generate a normal to it to define a plane parallel to the normal
    plane_vec = base_coords[0]-base_cen
    plane_vec = plane_vec/np.sqrt(np.sum(plane_vec**2))
    slice_norm = np.cross(base_norm,plane_vec)

    #Save a dotted line representing the initial pointing vector
    pts_pnv = [COM + 0.2*i*plane_vec for i in range(50)]
    save_points_bare(np.array(pts_pnv),"first_pointing")

    #Calculate a root transform to rotate the parallel plane around the base normal
    C = np.array([  [              0, -1*base_norm[2],    base_norm[1]],
                    [   base_norm[2],               0, -1*base_norm[0]],
                    [-1*base_norm[1],    base_norm[0],              0]])

    #Calculate all the relative vectors to the COM
    rel_vecs = centers-COM
    rel_units = rel_vecs/np.dot(np.array([np.sqrt(np.sum(rel_vecs**2,axis=1))]).T,np.ones((1,3)))

    #Iterate the parallel plane through 6.28 radians and check areas close to splitting the model in half
    #   that gives lines at the horizontal and lateral cut planes (left, right, forward and back on the model)
    #   we go for the back since it's unique, get all w/i 5% of a half-cut, and average to get the back pointing vector
    t = 0.0 #Angle to check
    min_sets = [] #list of ~1/2 slice angles
    min_slices = [] #Slice normals associated with ~1/2 cuts
    min_slice = np.zeros((1,3)) #normals for averaging
    min_ct = 0 #count for averaging
    while t < 6.28:
        #Finish the transform with the rotation angle
        Ra = np.identity(3) + C*math.sin(t) + np.dot(C,C)*(1-math.cos(t))

        #Transform the slicing plane
        new_slice_norm = np.dot(Ra,slice_norm)

        #Project onto the slice plane normal
        slice_proj = np.dot(rel_vecs,new_slice_norm)

        #Calculate ratio between 'left' and 'right' side area counts
        slice_pos = np.sum(1*(slice_proj > 0.0)*vp_mag)
        slice_neg = np.sum(1*(slice_proj < 0.0)*vp_mag)
        ratio = slice_pos/(slice_pos+slice_neg)

        #Threshold to w/in 1/2 ratio to count a slice as a valid 1/2 cut
        p_th = HALF_CUT_RATIO
        if 0.5-p_th < ratio < 0.5+p_th:

            #If inside the threshold ratio, filter by whether there are points in a cone of the vector to see
            filt = np.sum(1.0*(np.arccos(np.dot(rel_units,new_slice_norm)) < CUT_SAMPLE_CONE_DEGREE*(3.1415/180.0)))

            #If there's no points (meaning it's pointing to the model 'back')
            if filt == 0:
                #Add it to the set of selected 'bbacl' vectors
                min_sets = min_sets + [t]
                min_slices = min_slices + [new_slice_norm]
                min_slice = min_slice + new_slice_norm
                min_ct+=1
        else:
            pass

        #rotate by 0.01 radians
        t+=0.01

    #Construct the pointing vector as -1x the average of the 'back' vectors
    v_point = -1*(min_slice/min_ct)[0][:]
    v_point = v_point/np.sqrt(np.sum(v_point**2)) #normalize it

    #Calculate the perpendicular vector as the cross product of the base normal and the new pointing vector
    v_perp = np.cross(v_point,base_norm)
    v_perp = v_perp/np.sqrt(np.sum(v_perp**2)) #Normalize it

    #Re-calculate the pointing vector to be *exactly* perpendicular to the perpendicular and the base normal
    v_point = np.cross(base_norm,v_perp)
    v_point = v_point/np.sqrt(np.sum(v_point**2)) #gotta make sure it's normalized

    #Make sure the base normal is precisely normalized
    base_norm = base_norm/np.sqrt(np.sum(base_norm**2))
    upper_norm = upper_norm/np.sqrt(np.sum(upper_norm**2))

    #Calculate the transforms from the computed axes
    if (_mode == "BOT"):
        TF_norm = base_norm
    if (_mode == "TOP"):
        TF_norm = upper_norm

    #Check if base norm is exactly perfect already - if not: math
    if not((TF_norm == np.array([[0,0,1]])).all()):
        #Calculate a vector to rotate the model Z to real Z (about perp to both via cross product)
        v_T1 = np.cross(TF_norm,np.array([[0,0,1]]))
        v_T1 = v_T1/np.sqrt(np.sum(v_T1**2)) #Normalize it

        #Calculate the angle to rotate by to get model Z to real Z and rotation transform
        t_T1 = np.arccos(np.dot(TF_norm,np.array([0,0,1])))
        C = np.array([  [            0, -1*v_T1[0][2],    v_T1[0][1]],
                        [   v_T1[0][2],             0, -1*v_T1[0][0]],
                        [-1*v_T1[0][1],    v_T1[0][0],            0]])
        T1 = np.identity(3) + C*math.sin(t_T1) + np.dot(C,C)*(1-math.cos(t_T1))
    else:
        #If the base normal is already exactly Z- just the identity. Cross product above breaks otherwise
        T1 = np.identity(3)

    #Transform all the axial vectors by the Z-fixing rotation
    v_point_T1 = np.dot(v_point,T1.T)
    v_perp_T1 = np.dot(v_perp,T1.T)
    base_norm_T1 = np.dot(TF_norm,T1.T)

    #Calculate transform to align non-Z axes to standard
    v_T2 = np.array([[0,0,1]]) #Just rotating around actual Z
    sgn = (v_point_T1[0]>0.0) - 1*(v_point_T1[0]<=0.0) #Calculate whether to go CW or CCW based on whether in Q1/2 or Q3/4
    t_T2 = (sgn)*np.arccos(np.dot(np.array([0,1,0]),v_point_T1))
    C = np.array([  [            0, -1*v_T2[0][2],    v_T2[0][1]],
                    [   v_T2[0][2],             0, -1*v_T2[0][0]],
                    [-1*v_T2[0][1],    v_T2[0][0],            0]])
    T2 = np.identity(3) + C*math.sin(t_T2) + np.dot(C,C)*(1-math.cos(t_T2))

    #Transform axial vectors again
    v_point_T12 = np.dot(v_point_T1,T2.T)
    v_perp_T12 = np.dot(v_perp_T1,T2.T)
    base_norm_T12 = np.dot(base_norm_T1,T2.T)

    #Calculate z-axis offset to put base of the model at 0 height (from base triangle to COM offset vector above)
    Tf3 = np.array([[0,0,1]])*v_bC_proj

    #Apply full transform to the mesh
    Tf_mesh_1 = np.dot(np.dot(_mesh[:,:3]-COM,T1.T),T2.T)+Tf3
    Tf_mesh_2 = np.dot(np.dot(_mesh[:,3:6]-COM,T1.T),T2.T)+Tf3
    Tf_mesh_3 = np.dot(np.dot(_mesh[:,6:9]-COM,T1.T),T2.T)+Tf3

    #Check for 0-crossing discrepancy (round errors) and offset tf vector
    Tf3_offs = min([np.min(Tf_mesh_1[:,2]),np.min(Tf_mesh_2[:,2]),np.min(Tf_mesh_3[:,2])])
    Tf3 = Tf3 - Tf3_offs*np.array([[0,0,1]])
    
    #Caculate X-offset value for key object (items placed at the base of the body) placement
    n_cens = (Tf_mesh_1+Tf_mesh_2+Tf_mesh_3)/3.0
    cen_sel = 1*(n_cens[:,0]>=-1*CENTER_SELECT_DIST)*(n_cens[:,0]<=CENTER_SELECT_DIST)
    cen_xs = n_cens[cen_sel==1,1]
    x_min = np.min(cen_xs)
    key_offs = KEY_SPACE_FACTOR*(x_min-KEY_OFFSET) + KEY_SPACE_OFFSET
    
    Tf3 = Tf3 - key_offs*np.array([[0,1,0]])

    #Offset the transformed matrices
    Tf_mesh_1 = Tf_mesh_1 - Tf3_offs*np.array([[0,0,1]]) - key_offs*np.array([[0,1,0]])
    Tf_mesh_2 = Tf_mesh_2 - Tf3_offs*np.array([[0,0,1]]) - key_offs*np.array([[0,1,0]])
    Tf_mesh_3 = Tf_mesh_3 - Tf3_offs*np.array([[0,0,1]]) - key_offs*np.array([[0,1,0]])

    #A whole diagnostic bunch of drawn lines
    pts = []
    pts1 = []
    pts2 = []
    pts3 = []
    pts4 = []

    #Draw the slice normals selected for the 'back' average vector
    for sl in min_slices:
        pts = pts + [COM + (sl)*(i/40)*10 for i in range(40)]

    #Draw the triangle defining the base plane (corner to corner and center to corner lines both)
    pts = pts + [base_coords[0] + (base_coords[1]-base_coords[0])*(i/40) for i in range(40)]
    pts = pts + [base_coords[1] + (base_coords[2]-base_coords[1])*(i/40) for i in range(40)]
    pts = pts + [base_coords[2] + (base_coords[0]-base_coords[2])*(i/40) for i in range(40)]
    pts = pts + [base_cen + (base_coords[0]-base_cen)*(i/40) for i in range(40)]
    pts = pts + [base_cen + (base_coords[1]-base_cen)*(i/40) for i in range(40)]
    pts = pts + [base_cen + (base_coords[2]-base_cen)*(i/40) for i in range(40)]

    #Draw the inital coordinate axes derived from the base plane (prior to slicing) at the base center
    pts1 = pts1 + [base_cen + plane_vec*(i/40)*5 for i in range(40)]
    pts1 = pts1 + [base_cen + slice_norm*(i/40)*5 for i in range(40)]
    pts1 = pts1 + [base_cen + base_norm*(i/40)*5 for i in range(40)]

    #Draw the calculated model axes (post slicing)
    pts2 = pts2 + [COM + v_point*(i/40)*10 for i in range(40)]
    pts2 = pts2 + [COM + v_perp*(i/40)*10 for i in range(40)]
    pts2 = pts2 + [COM + base_norm*(i/40)*10 for i in range(40)]

    #Draw the final transformed axes at (0,0,0)
    pts3 = pts3 + [base_norm_T12*(i/40)*10 for i in range(40)]
    pts3 = pts3 + [v_point_T12*(i/40)*10 for i in range(40)]
    pts3 = pts3 + [v_perp_T12*(i/40)*10 for i in range(40)]

    #A second points file plotting the transformed point centers for the whole model
    pts4 = np.dot(np.dot(centers-COM,T1.T),T2.T)+Tf3

    #Save all the diagnostic lines
    save_points_bare(np.array(pts),"diag1")
    save_points_bare(np.array(pts1),"diag2")
    save_points_bare(np.array(pts2),"diag3")
    save_points_bare(np.array(pts3),"diag4")
    save_points_bare(np.array(pts4),"diag5")

    base_triangle = [base_norm,base_coords,base_cen]
    upper_triangle = [upper_norm,upper_coords,upper_cen]

    #Final output:
    #  v_point,v_perp,base_norm: Calculated axes for the model frame
    #  [pts,pts2]: Diagnostic points list for drawing salient calculation marks
    #  [T1,T2,Tf3]: Final computed transforms applied to align the model
    #  [Tf_mesh_1,Tf_mesh_2,Tf_mesh_3]: Transformed actual mesh points for conversion into an STL
    #  [COM,base_triangle,upper_triangle]: Calculated bounding parameters of the model itself
    return v_point,v_perp,base_norm,[pts,pts2],[T1,T2,Tf3],[Tf_mesh_1,Tf_mesh_2,Tf_mesh_3],[COM,base_triangle,upper_triangle,init_tri_pts]


####
#Cut/Etrude
####

def find_base_sal(_mesh):
    #Function to extract model base
    #   Works by taking a reference vector, and finding all face angle to it
    #   selects out most common angled normal weighted by face area
    #   This gets a set of faces all angled in common direction
    #   From there, checks distance along face normals relative to COM
    #   of theses, selects face set at most common projection distance
    #Returns list of indices, center points, and faces of base

    #Fresh copy for output
    mesh_out = np.copy(_mesh)

    #Extract mesh points
    mesh_pt1 = _mesh[:,:3]
    mesh_pt2 = _mesh[:,3:6]
    mesh_pt3 = _mesh[:,6:9]

    #get triangle centers
    centers = np.zeros(np.shape(mesh_pt1))
    centers[:,0] = (mesh_pt1[:,0]+mesh_pt2[:,0]+mesh_pt3[:,0])/3
    centers[:,1] = (mesh_pt1[:,1]+mesh_pt2[:,1]+mesh_pt3[:,1])/3
    centers[:,2] = (mesh_pt1[:,2]+mesh_pt2[:,2]+mesh_pt3[:,2])/3

    #Get vectors from triangle sides, normals, and magnitudes
    v1 = mesh_pt1 - mesh_pt2
    v2 = mesh_pt3 - mesh_pt2
    vp = np.cross(v1,v2)
    vp_mag = np.sqrt(np.sum(vp**2,axis=1))

    #Get weighted centers
    cen_a = np.copy(centers)
    cen_a[:,0] = cen_a[:,0]*vp_mag
    cen_a[:,1] = cen_a[:,1]*vp_mag
    cen_a[:,2] = cen_a[:,2]*vp_mag
    COM = np.sum(cen_a,axis=0)/np.sum(vp_mag)

    #Make a reference vector for clustering (<0,0,1> for now)
    v_ref = np.zeros((1,3))
    v_ref[0,2] = 1

    #Calculate angles between all extant vectors and reference
    z_angles_cos = np.dot(vp,v_ref.T)[:,0]/vp_mag

    #Get histogram of angles weighted by face area
    hist,bins = np.histogram(z_angles_cos,bins=FIRST_ANGLE_BINS,weights=vp_mag)

    #Select out face normal angle range w/ reatest associated total area
    max_bin = np.argmax(hist)
    max_range = (bins[max_bin],bins[max_bin+1])

    #First filter- faces with normal angle to reference w/i max range
    v_filt_1 = 1*(max_range[0] <= z_angles_cos)*(z_angles_cos <= max_range[1])

    save_points_bare(centers[v_filt_1==1],"Base_f_1")

    #Calculate indices, vectors, magnitude and points for filtered faces
    vf_1_ind = np.array([range(np.shape(v_filt_1)[0])])[0,v_filt_1==1]
    vf_1_vecs = vp[v_filt_1==1]
    vf_1_mag = np.sqrt(np.sum(vf_1_vecs**2,axis=1))
    vf_1_pts = centers[v_filt_1==1,:]

    #Grab a representative face vector w/i selected set
    first_vf1 = np.argmax(v_filt_1)
    v_prim = vp[first_vf1,:]
    v_prim = v_prim/np.sqrt(np.sum(v_prim*v_prim)) #normalize that vector

    #Get vectors from COM to selected faces
    v_COM = vf_1_pts - COM

    #Calculate the projection of distance from faces parallel to normal vectors
    axis_dists = np.dot(v_COM,v_prim)

    #Get histogram of axial projections (ranged from -10 to 10 distance from COM)
    hist,bins =  np.histogram(axis_dists,bins=SECOND_ANGLE_BINS,range=[-10.0,10.0])

    #Grab the most common distance range
    max_bin = np.argmax(hist)
    max_range = (bins[max_bin],bins[max_bin+1])

    #Second filter is set of faces near most common axial distance
    v_filt_2 = 1*(max_range[0] <= axis_dists)*(axis_dists <= max_range[1])

    save_points_bare(centers[v_filt_1==1][v_filt_2==1],"Base_f_2")

    #Construct output- indices, points, and faces at most common axial dist
    vf_fin_ind = vf_1_ind[v_filt_2==1]
    vf_fin_pts = vf_1_pts[v_filt_2==1,:]
    vf_fin_faces = mesh_out[vf_fin_ind,:]

    #Vectors from COM
    v_com_1 = mesh_pt1 - COM
    v_com_2 = mesh_pt2 - COM
    v_com_3 = mesh_pt3 - COM

    #Project onto axial vector
    axial_full_1 = np.dot(v_com_1,v_prim)
    axial_full_2 = np.dot(v_com_2,v_prim)
    axial_full_3 = np.dot(v_com_3,v_prim)

    #Calculate the width around the largest bin area
    span = max_range[1]-max_range[0]

    #Find the points inside that largest bin
    axial_vf_1 = 1*(max_range[0]-1.0*span <= axial_full_1)*(axial_full_1 <= max_range[1]+1.0*span)
    axial_vf_2 = 1*(max_range[0]-1.0*span <= axial_full_2)*(axial_full_2 <= max_range[1]+1.0*span)
    axial_vf_3 = 1*(max_range[0]-1.0*span <= axial_full_3)*(axial_full_3 <= max_range[1]+1.0*span)

    #Grab the points inside that selection
    pts_strp_1 = mesh_pt1[axial_vf_1 == 1]
    pts_strp_2 = mesh_pt2[axial_vf_2 == 1]
    pts_strp_3 = mesh_pt3[axial_vf_3 == 1]

    #Merge the point components by triangle
    pts_strp = np.concatenate([pts_strp_1,pts_strp_2,pts_strp_3])

    #Save the point selection
    save_points_bare(np.array(pts_strp),"Base_fin")

    #Return the final indices, points, and faces of the selected base, the base points
    #set, and the points selected out
    return vf_fin_ind,vf_fin_pts,vf_fin_faces,[axial_vf_1,axial_vf_2,axial_vf_3],pts_strp

def fix_flat(_mesh, base_norm,h_th,offset = None):
    #Utility function to eliminate base irregularities along the base norm
    #   *Forcefully flattens all points on the mesh w/i h_th of the lowest point along
    #   base_norm to be at the same 'height' along base norm

    #get mesh points
    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    #Project the points onto the base norm vector
    base_proj_1 = np.dot(mesh_pt1_rot,base_norm)
    base_proj_2 = np.dot(mesh_pt2_rot,base_norm)
    base_proj_3 = np.dot(mesh_pt3_rot,base_norm)

    #Get the max and min projection lengths
    min_proj = min([np.min(base_proj_1),np.min(base_proj_2),np.min(base_proj_3)])
    max_proj = max([np.max(base_proj_1),np.max(base_proj_2),np.max(base_proj_3)])
    total_height = max_proj - min_proj

    #Get the regions to flatten (within the threshold of the lowest base region)
    flat_reg_1 = 1*(base_proj_1 < min_proj + h_th)
    flat_reg_2 = 1*(base_proj_2 < min_proj + h_th)
    flat_reg_3 = 1*(base_proj_3 < min_proj + h_th)

    #Select out the base points 
    base_sel = []
    base_sel = base_sel + list(mesh_pt1_rot*np.dot(np.array([flat_reg_1]).T,np.ones((1,3))))
    base_sel = base_sel + list(mesh_pt2_rot*np.dot(np.array([flat_reg_2]).T,np.ones((1,3))))
    base_sel = base_sel + list(mesh_pt3_rot*np.dot(np.array([flat_reg_3]).T,np.ones((1,3))))

    #Get the distances by which the current base violates the flatness threshold
    d_th_1 = (base_proj_1 - min_proj)*flat_reg_1
    d_th_2 = (base_proj_2 - min_proj)*flat_reg_2
    d_th_3 = (base_proj_3 - min_proj)*flat_reg_3

    #Calculate the local shift to correct the threshold violation
    d_shift_1 = base_norm*np.dot(np.array([d_th_1]).T,np.ones((1,3)))
    d_shift_2 = base_norm*np.dot(np.array([d_th_2]).T,np.ones((1,3)))
    d_shift_3 = base_norm*np.dot(np.array([d_th_3]).T,np.ones((1,3)))

    #Shift the base by the amount to flatten it
    mesh_pt1_rot_o = mesh_pt1_rot - d_shift_1
    mesh_pt2_rot_o = mesh_pt2_rot - d_shift_2
    mesh_pt3_rot_o = mesh_pt3_rot - d_shift_3

    #If there is a known offset to apply
    if offset != None:
        #apply it to all the flattened regions
        h_th_1 = offset*flat_reg_1
        h_th_2 = offset*flat_reg_2
        h_th_3 = offset*flat_reg_3

        #Calculate the offset shift
        h_shift_1 = base_norm*np.dot(np.array([h_th_1]).T,np.ones((1,3)))
        h_shift_2 = base_norm*np.dot(np.array([h_th_2]).T,np.ones((1,3)))
        h_shift_3 = base_norm*np.dot(np.array([h_th_3]).T,np.ones((1,3)))    

        #Apply the offset to the mesh directly
        mesh_pt1_rot_o = mesh_pt1_rot_o + h_shift_1
        mesh_pt2_rot_o = mesh_pt2_rot_o + h_shift_2
        mesh_pt3_rot_o = mesh_pt3_rot_o + h_shift_3

    #Make a copy for output and populate the new values
    out_mesh = copy.deepcopy(_mesh)
    out_mesh[:,:3] = mesh_pt1_rot_o
    out_mesh[:,3:6] = mesh_pt2_rot_o
    out_mesh[:,6:9] = mesh_pt3_rot_o

    #Return the output mesh and the selected base points
    return out_mesh,base_sel

def fix_flat2(mod,height):
    # Functions to forceably assign a rough-cut base at 0-height to be actually
    #   precisely flat. Works by translating to a given offset height, and then
    #   selecting all points thereafter in the bases below the 0-height threshold,
    #   assigning those faces to have a z value of 0

    #Copy the mesh to base flatten
    CE_mesh = np.copy(mod._mesh)

    #Calculate the height of the model
    mod_height = max([np.max(CE_mesh[:,2]),np.max(CE_mesh[:,5]),np.max(CE_mesh[:,8])])

    #Move the model down by the requisite height
    CE_mesh[:,2] = CE_mesh[:,2] + (height-mod_height)
    CE_mesh[:,5] = CE_mesh[:,5] + (height-mod_height)
    CE_mesh[:,8] = CE_mesh[:,8] + (height-mod_height)

    #Get the base points
    base_ind,base_pts,base_faces,ax_vf,strp_pts = find_base_sal(CE_mesh)

    #set the base points' z values to 0
    CE_mesh[ax_vf[0]==1,2] = 0.0
    CE_mesh[ax_vf[1]==1,5] = 0.0
    CE_mesh[ax_vf[2]==1,8] = 0.0

    #find all non-base points which are still negative
    v1_zero_cross = 1*(CE_mesh[:,2] < 0.0)
    v2_zero_cross = 1*(CE_mesh[:,5] < 0.0)
    v3_zero_cross = 1*(CE_mesh[:,8] < 0.0)

    #Project them to zero, too
    CE_mesh[v1_zero_cross==1,2] = 0.0
    CE_mesh[v2_zero_cross==1,5] = 0.0
    CE_mesh[v3_zero_cross==1,8] = 0.0

    #return the new mesh and ID'd base points
    return [CE_mesh[:,:3],CE_mesh[:,3:6],CE_mesh[:,6:9]],strp_pts

def base_cut_min(fil,min_thresh=0.1,depth = 10.0):
    #Function to cut a model around a plane

    #Make a pv mesh of the file
    p_mesh = pv.PolyData(fil)

    #Get the X, Y, and Z extent of the mesh
    x_min = min(p_mesh.points[:,0])
    x_max = max(p_mesh.points[:,0])

    y_min = min(p_mesh.points[:,1])
    y_max = max(p_mesh.points[:,1])

    z_min = min(p_mesh.points[:,2])
    z_max = max(p_mesh.points[:,2])

    #loop through a sequence of slices along the Z axis
    h = 0.0
    sx_ratio = 0.0
    while sx_ratio < 0.9:
        h+=0.01 #Increment by 0.01 height units
        p_slice = p_mesh.slice(normal=(0,0,1),origin=((x_min+x_max)/2.0,(y_min+y_max)/2.0,z_min+h))
        if np.shape(p_slice.points)[0] != 0: #calculate the min and max slice coordinates
            sx_min = min(p_slice.points[:,0])
            sx_max = max(p_slice.points[:,0])
            sx_ratio = (sx_max-sx_min)/(x_max-x_min) #calculate the ratio of the extent to the maximum

    #calculate a center for the placement of the slice removal object
    _center = ((x_min+x_max)/2.0,(y_min+y_max)/2.0,z_min-(depth/2.0)+h)

    #Construct a 'cube' guaranteed to encompass all the area to be removed
    the_CUBE = pv.Cube(center=_center, x_length= 1.15*(x_max-x_min), y_length=1.15*(y_max-y_min), z_length=depth)
    the_CUBE = the_CUBE.triangulate() #Make it a triangle mesh

    #Remove the 'cube' from the mesh
    sliced = p_mesh.boolean_difference(the_CUBE)
    sliced.clean(inplace=True) #clean the object left over

    sliced.save("sliced "+fil.split("\\")[-1]) #Save the output cut object

    #Return the sliced body
    return sliced

def cut(_mesh, base_norm, height):
    #Wrapper function to execute a base cut
    return fix_flat(_mesh, base_norm,0.25,offset = -1*height)

def extrude(_mesh, base_norm, height):
    #Wrapper function to execute a base extrusion
    return fix_flat(_mesh, base_norm,0.25,offset = height)

####
#Transforms
####

def apply_TF(_mesh,COM,TF):
    #Utility to apply a transform to an arbitrary mesh

    #Extract transform components
    T1 = TF[0]
    T2 = TF[1]
    Tf3 = TF[2]

    #Apply full transform to the mesh
    Tf_mesh_1 = np.dot(np.dot(_mesh[:,:3]-COM,T1.T),T2.T)+Tf3
    Tf_mesh_2 = np.dot(np.dot(_mesh[:,3:6]-COM,T1.T),T2.T)+Tf3
    Tf_mesh_3 = np.dot(np.dot(_mesh[:,6:9]-COM,T1.T),T2.T)+Tf3

    #Check for 0-crossing discrepancy (round errors) and offset tf vector
    Tf3_offs = min([np.min(Tf_mesh_1[:,2]),np.min(Tf_mesh_2[:,2]),np.min(Tf_mesh_3[:,2])])

    #Offset the transformed matrices
    Tf_mesh_1 = Tf_mesh_1 - Tf3_offs*np.array([[0,0,1]])
    Tf_mesh_2 = Tf_mesh_2 - Tf3_offs*np.array([[0,0,1]])
    Tf_mesh_3 = Tf_mesh_3 - Tf3_offs*np.array([[0,0,1]])

    return [Tf_mesh_1,Tf_mesh_2,Tf_mesh_3]

def get_RPY(R,P,Y):
    #Calculate a set of RPY transforms from angles

    #Roll X
    ct = math.cos(R)
    st = math.sin(R)
    Roll_M = np.array([[1,0,0],[0,ct,-1*st],[0,st,ct]])

    #Pitch Y
    ct = math.cos(P)
    st = math.sin(P)
    Pitch_M = np.array([[ct,0,st],[0,1,0],[-1*st,0,ct]])

    #Yaw Z
    ct = math.cos(Y)
    st = math.sin(Y)
    Yaw_M = np.array([[ct,-1*st,0],[st,ct,0],[0,0,1]])

    return Roll_M,Pitch_M,Yaw_M

def translate(_mesh,xM,yM,zM):
    # Little function to apply a space transformation vis-a-vis the COM

    #Grab the points
    mesh_tf_1 = _mesh[:,:3]
    mesh_tf_2 = _mesh[:,3:6]
    mesh_tf_3 = _mesh[:,6:9]

    #Move the x points
    mesh_tf_1[:,0] = mesh_tf_1[:,0] + xM
    mesh_tf_2[:,0] = mesh_tf_2[:,0] + xM
    mesh_tf_3[:,0] = mesh_tf_3[:,0] + xM

    #Move the y points
    mesh_tf_1[:,1] = mesh_tf_1[:,1] + yM
    mesh_tf_2[:,1] = mesh_tf_2[:,1] + yM
    mesh_tf_3[:,1] = mesh_tf_3[:,1] + yM

    #Move the z points    
    mesh_tf_1[:,2] = mesh_tf_1[:,2] + zM
    mesh_tf_2[:,2] = mesh_tf_2[:,2] + zM
    mesh_tf_3[:,2] = mesh_tf_3[:,2] + zM

    #Return the mesh arrays
    return [mesh_tf_1,mesh_tf_2,mesh_tf_3]

####
#Load/Save
####

def make_asc(name,data):
    #A helper function to make an output ASC file to visualize points in a CAD viewer

    N,d = data.shape
    f_out = open(name.split(".")[0].split("\\")[-1]+"_pts.asc",'w')
    for i in range(N):
        o_str = ""
        for j in range(d):
            o_str = o_str + str(data[i,j]) + " "
        o_str = o_str[:-1]+"\n"
        f_out.write(o_str)
    f_out.close()


def save_mesh_as(o_mesh,n_arrays,name):
    #A cheap function to save an output mesh object

    if len(n_arrays)!=3:
        n_arrays.save(name)
    else:
        o_mesh[:,:3] = n_arrays[0]
        o_mesh[:,3:6] = n_arrays[1]
        o_mesh[:,6:9] = n_arrays[2]
        o_mesh.save(name)
    return 1

def save_points_bare(pts,name):
    #Wrapper to make points into an output file for viewing
    make_asc(name,np.array(pts))

def save_points_as(pts_list,name):
    # Utility to make an ACS specifically from a list of points 

    op_axis = []
    for i in range(10):
        #print(i*pts_list[0]+pts_list[3])
        op_axis = op_axis + [i*pts_list[0]+pts_list[3],i*pts_list[1]+pts_list[3],i*pts_list[2]+pts_list[3]]

    make_asc(name,np.array(op_axis))

def save_line_as(ot,pt,name):
    # utility to generate an ASC of a line between two points
    vl = pt-ot
    op_axis = []
    for i in range(51):
        op_axis = op_axis + [ot + (i*vl)/50]

    make_asc(name,np.array(op_axis))

####
#Analytics
####

def fuzzy_model(_mesh,filt=None):
    # Function to create vector lines at the centers of face triangles, along the
    #   normal of that face

    if type(filt) == type(None):
        filt = np.ones((np.shape(_mesh)[0],3))

    #get mesh points
    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    #calculate triangle centers
    centers = np.zeros(np.shape(mesh_pt1_rot))
    centers[:,0] = (mesh_pt1_rot[:,0]+mesh_pt2_rot[:,0]+mesh_pt3_rot[:,0])/3
    centers[:,1] = (mesh_pt1_rot[:,1]+mesh_pt2_rot[:,1]+mesh_pt3_rot[:,1])/3
    centers[:,2] = (mesh_pt1_rot[:,2]+mesh_pt2_rot[:,2]+mesh_pt3_rot[:,2])/3

    #Calculate patch areas
    v1 = mesh_pt1_rot - mesh_pt2_rot
    v2 = mesh_pt3_rot - mesh_pt2_rot
    vp = np.cross(v1,v2)
    vp_mag = np.sqrt(np.sum(vp**2,axis=1))
    v_vecs = -1*vp/np.dot(np.array([vp_mag]).T,np.ones((1,3)))

    #Make a points list and select the centers and vectors to plot
    pts = []
    cens_f = centers[filt==1]
    vecs_f = v_vecs[filt==1]

    #Construct the points for each vector set
    for a in range(len(cens_f)):
        pts = pts + [cens_f[a] + vecs_f[a]*(b/5) for b in range(5)]

    #Return the vector plot points
    return pts

def check_extent_z(_mesh,COM,z_min,z_max):
    # Function to determine the height offset at which the model's base
    #   profile is entirely intersecting the XY plane (i.e. the cut height to
    #   slice to make the base flat against small tilts)

    #Grab the triangle points
    mesh_pt1_rot = _mesh[0]
    mesh_pt2_rot = _mesh[1]
    mesh_pt3_rot = _mesh[2]

    #get triangle centers
    centers = np.zeros(np.shape(mesh_pt1_rot))
    centers[:,0] = (mesh_pt1_rot[:,0]+mesh_pt2_rot[:,0]+mesh_pt3_rot[:,0])/3
    centers[:,1] = (mesh_pt1_rot[:,1]+mesh_pt2_rot[:,1]+mesh_pt3_rot[:,1])/3
    centers[:,2] = (mesh_pt1_rot[:,2]+mesh_pt2_rot[:,2]+mesh_pt3_rot[:,2])/3

    #Calcualte COM-relative vectors
    vecs = centers-COM

    #Figure the range of z spacing, set the examination height, and the
    #   resolution of checks (0.005 mm is 1/2 the min resolution of the mill
    #   so Nyquist guarantee to find real minimum)
    h_ran = z_max-z_min
    z_samp = z_min
    z_res = 0.005

    #Ranges of counted points
    ct_spans = []

    #Loop over heights, up to w/i 15% of the range (assumes 'bottom' will be
    #   within that range)
    k=0
    while z_samp < z_min + 0.15*h_ran:

        # grab all the points inside the resolution height about the current
        #    sample height
        z_sel = 1*(np.abs(centers[:,2]-z_samp) < z_res*h_ran)

        # Grab the x and y coordinates which are in the selected area
        x_sel = centers[:,0]*z_sel
        y_sel = centers[:,1]*z_sel

        # Get the extent of the spanned selected points, with the non-inculded points
        #   left out (so all negative ranges don't have any zkew from 0s, eg.)
        xs_min = np.min(x_sel + 10000*(1-z_sel))
        xs_max = np.max(x_sel - 10000*(1-z_sel))
        ys_min = np.min(y_sel + 10000*(1-z_sel))
        ys_max = np.max(y_sel - 10000*(1-z_sel))
        num_pt = np.sum(z_sel)

        # Record the maximum extents for this slice        
        ct_spans = ct_spans + [(xs_max-xs_min,ys_max-ys_min)]

        #Optional diagnostic
        #print(k,xs_max-xs_min,ys_max-ys_min,num_pt)

        #Increase the sample height
        z_samp = z_samp + z_res*h_ran
        k+=1

    # Set the max search span dimensions to initial extremes
    ct_x = -1
    ct_x_e = 9999
    ct_y = -1
    ct_y_e = 9999

    # Loop over the gathered spans and check the height at which the extents remain stable
    for i in range(len(ct_spans))[1:]:

        # Grab the average spacing between the current element and the last one for the x component of the points
        m1_x = (ct_spans[i][0]-ct_spans[0][0])/(i)
        m2_x = (ct_spans[-1][0]-ct_spans[i][0])/(i)

        # Get the sum of all the ranges from either tail (left/right or front/back) of the spans
        e1_x = sum([abs(ct_spans[j][0]-(ct_spans[0][0]+m1_x*j)) for j in range(i)])
        e2_x = sum([abs(ct_spans[j+i][0]-(ct_spans[i][0]+m2_x*(j))) for j in range(len(ct_spans)-i)])
        e_x = e1_x+e2_x #Make the total count at both ends

        #If it's a new minimum, save it, and the index
        if e_x < ct_x_e:
            ct_x_e = e_x
            ct_x = i

        #Do the same as above for the y dimensions
        m1_y = (ct_spans[i][1]-ct_spans[0][1])/(i)
        m2_y = (ct_spans[-1][1]-ct_spans[i][1])/(i)
        e1_y = sum([abs(ct_spans[j][1]-(ct_spans[0][1]+m1_y*j)) for j in range(i)])
        e2_y = sum([abs(ct_spans[j+i][1]-(ct_spans[i][1]+m2_y*(j))) for j in range(len(ct_spans)-i)])
        e_y = e1_y+e2_y
        if e_y < ct_y_e:
            ct_y_e = e_y
            ct_y = i

    #Grab the maximum cutoff count index between x and y dimensions
    cutoff = max([ct_x,ct_y])
    cutoff_h = cutoff*z_res #Set the actual height to the index*the resolution

    return cutoff_h

def get_COM(_mesh,_name=""):
    #Grab the COM of a body (weighted by patch area)

    #get mesh points
    mesh_pt1_rot = _mesh[:,:3]
    mesh_pt2_rot = _mesh[:,3:6]
    mesh_pt3_rot = _mesh[:,6:9]

    #Calculate patch areas
    v1 = mesh_pt1_rot - mesh_pt2_rot
    v2 = mesh_pt3_rot - mesh_pt2_rot
    vp = np.cross(v1,v2)
    vp_mag = np.sqrt(np.sum(vp**2,axis=1))

    #Get weighted centers
    cen_a = np.copy(mesh_pt1_rot)
    cen_a[:,0] = cen_a[:,0]*vp_mag
    cen_a[:,1] = cen_a[:,1]*vp_mag
    cen_a[:,2] = cen_a[:,2]*vp_mag
    COM = np.sum(cen_a,axis=0)/np.sum(vp_mag)

    return COM

def get_recti_box(_mesh):
    # Grap the rectilinear frame-fixed bounds for a model

    #Grab min extent in XYZ
    min_x = min([np.min(_mesh[0][:,0]),np.min(_mesh[1][:,0]),np.min(_mesh[2][:,0])])
    min_y = min([np.min(_mesh[0][:,1]),np.min(_mesh[1][:,1]),np.min(_mesh[2][:,1])])
    min_z = min([np.min(_mesh[0][:,2]),np.min(_mesh[1][:,2]),np.min(_mesh[2][:,2])])

    #Grab max extent in XYZ
    max_x = max([np.max(_mesh[0][:,0]),np.max(_mesh[1][:,0]),np.max(_mesh[2][:,0])])
    max_y = max([np.max(_mesh[0][:,1]),np.max(_mesh[1][:,1]),np.max(_mesh[2][:,1])])
    max_z = max([np.max(_mesh[0][:,2]),np.max(_mesh[1][:,2]),np.max(_mesh[2][:,2])])

    #Return the boundary corners
    return min_x,max_x,min_y,max_y,min_z,max_z


####
#Model Healing
####

def check_NM_errors(fil):
    #Function to search for non-manifold errors

    #Load the STL as a triangle mesh in Open3D
    p_mesh = o3d.io.read_triangle_mesh(fil)
    p_mesh.compute_vertex_normals()
    p_mesh.remove_duplicated_vertices()

    #Grab a list of non-manifold edges & vertices
    nme_list = p_mesh.get_non_manifold_edges(allow_boundary_edges=True)
    nmv_list = p_mesh.get_non_manifold_vertices()

    #Get the length of the list
    nme_count = len(nme_list)
    nmv_count = len(nmv_list)

    #flag for if there's nme errors, and non-correspondind vertices
    crit = not((nme_count > 0) and (nme_count < nmv_count))

    #Return flag, file, and counts of nme and nmv 
    return crit,fil,nme_count,nmv_count

def check_and_correct_NM(fil,ct_max=10):
    # Function to find and correct non-manifold errors

    #Find the nmes, first
    crit,_,nme,nmv = check_NM_errors(fil)

    #Until the errors are removed, or the attempt limit is hit
    _ct = 1
    while crit and _ct<ct_max:
        _ct+=1

        #Try cleaning the stl file
        fil_n = clean_stls(fil,lead_string="f"+str(_ct))

        #Check for errors again
        crit,_,nme_n,nmv_n = check_NM_errors(fil_n)

        #If it made it worse, just return the original
        if (nme_n >= nme) or (nmv >= nmv):
            return fil,nme,nmv
        #Otherwise update and keep cleaning
        else:
            fil = fil_n
            nme = nme_n
            nmv = nmv_n

    #Return the final cleaned file
    return fil

def clean_stls(fil,lead_string="cleaned ",do_o3d=True,do_pv=True,hole_size=1000):
    #Function to pass STLS through all of the repair functions thus found in open3D and VTK

    if do_o3d:
        #Optional open3D cleans

        #Read in in o3d format
        p_mesh = o3d.io.read_triangle_mesh(fil)

        #Run through all the repairs
        p_mesh.remove_degenerate_triangles()
        p_mesh.remove_duplicated_triangles()
        p_mesh.remove_duplicated_vertices()
        p_mesh.remove_non_manifold_edges()

        #Necessary rebuild to write the files
        p_mesh.compute_vertex_normals()

        if not(do_pv):
            #Write off final file if not doing VTK fixes
            o3d.io.write_triangle_mesh(lead_string+fil.split("\\")[-1], p_mesh)
        else:
            #Write temp file if moving on to VTK
            o3d.io.write_triangle_mesh("temp"+fil.split("\\")[-1], p_mesh)

    if do_pv:
        #Optional VTK cleans
        
        if not(do_o3d):
            #If not running the o3d fixes, open the original file
            p_mesh = pv.PolyData(fil)
        else:
            #If o3d doing, open temp file
            p_mesh = pv.PolyData("temp"+fil.split("\\")[-1])

        #Do VTK fixes
        p_mesh = p_mesh.fill_holes(hole_size)

        #Safe final file
        p_mesh.save(lead_string+fil.split("\\")[-1])
        return lead_string+fil.split("\\")


#Main test loop
if __name__ == '__main__':

    #Grab the sample file set (from the current directory, now)
    samples_file = ""
    samples = glob.glob(samples_file+"*.stl")

    #Diagnostic counting and timing variables
    cnt = 0
    ti = time.time()

    # Data collection for analytics
    total_data = []
    sample_slices = []

    #Run through each sample
    for samp in samples:

        #Output the name of the file
        print(" ")
        print(samp)

        #Load the mesh in
        _mesh = mesh.Mesh.from_file(samp)

        #Optional downsample- should be done at another step
        #_,_,_,_ = downsample(samp,D_SAMP=60000) #Optional downsample
        #_ = clean_stls(samp,lead_string="cleaned ",do_o3d=True,do_pv=True,hole_size=1000)

        t1 = time.time()

        #NEW getting of the main axes, AND transforming the model
        print("Acquire coodrinate system")
        prime,horiz,vert,pts_aln,TF,_,bound = get_principal_axis2(_mesh,_name="")
        print(prime,horiz,vert)
        print("dT: ",time.time()-t1)
        t1 =time.time()
        print("________")

        COM = bound[0]
        base_tri = bound[1]
        upper_tri = bound[2]
        init_tri = bound[3]

        upper_cen = upper_tri[2]
        upper_coords = upper_tri[1]
        upper_norm = upper_tri[0]

        base_cen = base_tri[2]
        base_coords = base_tri[1]
        base_norm = base_tri[0]

        print("TRANSFORM MESH")
        print(TF)
        print("dT: ",time.time()-t1)
        t1 =time.time()
        print("________")

        #Apply the derived transform to the model
        out_mesh_fin = apply_TF(_mesh,COM,TF)

        print("Cut/Extrude")
        #Flatten Model base against base_norm
        flattened,base_pts = fix_flat2(_mesh,19.0)
        print("NM prior: ",check_NM_errors(samp))

        print(len(base_pts),np.histogram(base_pts,bins=50))
        print("dT: ",time.time()-t1)
        t1 =time.time()
        print("________")

        mesh_cut,base_pts = cut(_mesh, vert,1.0)
        mesh_ext,base_pts = extrude(_mesh, vert,1.0)
        mesh_cut.save(samp.split(".")[0]+"_cut.stl")
        mesh_ext.save(samp.split(".")[0]+"_ext.stl")
        
        print("NM post, cut: ",check_NM_errors(samp.split(".")[0]+"_cut.stl"))
        print("NM post, ext: ",check_NM_errors(samp.split(".")[0]+"_ext.stl"))

        print("dT: ",time.time()-t1)
        t1 =time.time()
        print("________")

        ####
        #DIAGNOSTIC OUTPUT SECTION
        ####

        flattened.save(samp.split(".")[0]+"_flatted.stl")
        save_points_bare(base_sel,"base_"+samp)

        ft = np.random.rand(np.shape(_mesh)[0])
        ft = 1*(ft < 0.33)
        fuzz = fuzzy_model(_mesh,filt = ft)

        #Make a new mesh object as a deep copy of the original
        new_mesh = copy.deepcopy(_mesh)
        new_mesh[:,:3] = out_mesh_fin[0]
        new_mesh[:,3:6] = out_mesh_fin[1]
        new_mesh[:,6:9] = out_mesh_fin[2]

        mesh_ext.save(samp.split(".")[0]+"_ext.stl")
        mesh_cut.save(samp.split(".")[0]+"_cut.stl")
        new_mesh.save(samp.split(".")[0]+"_FIN.stl")
        flattened.save(samp.split(".")[0]+"_Flat.stl")

        cnt+=1
        
    #Total execution time measurement
    t2 = time.time()

    #Print the final speed diagnostics
    print("--------")
    print("Runtime: ",round(t2-ti,2))
    print("Average per model: ",round((t2-ti)/cnt,2))





