###
#
# Script to batch check for NME errors in cad models, and sort them for repair or checking
#
###

#Standards
import math,time,random

#File handling
import glob

#Mesh processing
import numpy as np
import open3d as o3d
import pyvista as pv
import shutil,os

#Grab all STLs in this directory
fils = glob.glob("*.stl")

#If the sort folders aren't extant, create them
if not(os.path.exists("to_examine")):
    os.makedirs("to_examine")
if not(os.path.exists("known_bad")):
    os.makedirs("known_bad")

def fix_NM_errs(fil,lead_string="cleaned "):
    #A slightly hacky function that repairs by a downsample with
    # topology preservation

    p_mesh = pv.PolyData(fil)
    p_mesh = p_mesh.decimate_pro(.1,preserve_topology=True)
    p_mesh.save(lead_string+fil.split("\\")[-1])

    return lead_string+fil.split("\\")[-1]

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

#Reporting string
rep_str = ""

#For each found stl
for fil in fils:

    #Try to heal the NMEs, and check if it worked after
    fil,_,_ = check_and_correct_NM(fil)
    crit,_,nme_count,nmv_count = check_NM_errors(fil)

    #Diagnostic output
    print(fil)
    print("NME: ",nme_count)
    print("NMV: ",nmv_count)

    #Add to report string
    rep_str = rep_str + fil + " : " + str(nme_count) + " : " + str(nmv_count) + "\n"

    #If errors still, move to the examination folder
    if crit:
        shutil.move(fil,"to_examine/"+fil)
    #If NME/Vs found, move to the known bad folder
    else:
        shutil.move(fil,"known_bad/"+fil)


#Report file
report = open("report.txt",'w')
report.write(rep_str)
report.close()

#Say that you're done
print("Done")

