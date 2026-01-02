<img width="919" height="553" alt="Screenshot from 2026-01-02 13-35-29" src="https://github.com/user-attachments/assets/7b8e5c03-83e1-4da9-b7a9-fe71605cc06d" />


# CAD Utilities

This library is a collection of tools for manipulating CAD files, primarily STLs, at the programmatic level (which is to say, without recourse to a UI editor). It contains a set of utility functions to downsample, repair, transform, cut, extrude, and analyze mesh files.

The primary contribution utility is the `get_principal_axis2` function, which finds a set of orthogonal axes to an arbitrary model, using a sequence of gerometric descriptors to identify reliable primary axes within a model based on approximants of fully orderable metrics and with those to constructs a consistent orientation.

It also includes tools for posing models, as well as analysis tools to identify 'base' regions of the model by looking for the largest subset of co-aligned vectors in the mesh, as well as methods to efficiently implement numerically perfect basing, such as for 3D printing.

The sections of functions includes:

#### Processing Functions

`subsample_mesh` which selects a number of patch center points for a fast representative point cloud 
`downsample` which actually reduces the side of a mesh to a given point size 

#### AutoALign

`auto_center` to pose an STL in a standard layout with the COM at 0,0 and the lowest Z at 0
`auto_rotate_full` which orients the model around a standard axis based on the median orientation of position vectors
`get_principal_axis2`, mentioned above, which finds reliable axial reference frames for bodies

#### Cut/Etrude

`find_base_sal` a function to extract a model base by searching for the most common normal vector cluster in faces
`fix_flat` which removes base irregularity along a given normal axis supplied as the facing direction for the model 
`fix_flat2` which operates the same as `fix_flat`, but using an extracted base as the reference plane
`base_cut_min` which cuts a model across a given base plane and affixes it to that plane if too high or low

#### Transforms

`apply_TF` which applies an arbitrary transform of the structure `(ROLL,PITCH,OFFSET)`
`get_RPY`, a utility to calculate RPY transforms from angles
`translate` to shift the model in space

#### Load/Save

`make_asc` to create an asc points file for point clouds, lines, or other diagnostics 
`save_mesh_as` a function to convert a 9D points object to an STL mesh 
`save_points_as` utility to convert a points list into a viewable asc file 
`save_line_a` utility to make a line illustration between two points 

#### Analytics

`fuzzy_model` a diagnostic to draw face normals onto a mesh as dotted lines 
`check_extent_z` which determines a height offset at which the model's base plane fully intersects XY
`get_COM` calculate a face-area-weighted center of mass for the object
`get_recti_box` find the rectilinear bounding box for the model

#### Model Healin

`check_NM_errors` a function to find non-manifold errors and vertices
`check_and_correct_NM` which attempts to iteratively remove NMEs, if possible
`clean_stls` a function to apply several STL healing methods to repair NMEs, NMVs, inverted normals, remove duplicates and degenerate triangles, and close holes

