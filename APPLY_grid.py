# ----// APPLY GRID //-------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# NOTE: This function receives image height, image width, preferred grid cell height
#	and grid cell width, and calculates and returns grid cell count and grid cell
#	coordinates.

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def APPLY_grid(im_height, im_width, grid_height, grid_width):

	ax = 0
	ay = 0
	bx = 0
	by = 0
	cx = 0
	cy = 0
	dx = 0
	dy = 0

	hor_cell_count = 0 							
	# INITIALIZE GRID COUNT IN THE HORIZONTAL DIRECTION

	ver_cell_count = 0							
	# INITIALIZE GRID COUNT IN THE VERTICAL DIRECTION
	
	cell_count = 0								
	# INITIALIZE TOTAL GRID COUNT
	
	grid_coordinates = []
	# INITIALIZE GRID COORDINATES

	while ver_cell_count <= (im_height // grid_height):

		while hor_cell_count < (im_width // grid_width):
			ax = hor_cell_count * grid_width
			ay = ver_cell_count * grid_height
			bx = ax + grid_width
			by = ay
			cx = ax
			cy = ay + grid_height
			dx = bx
			dy = cy

			grid_coordinates.append([ax, ay, bx, cy])
			# CALCULATE ax, ay, bx, cy
			# CALCULATION OF by, cx, dx, AND dy IS NOT NECESSARY
			
			hor_cell_count += 1					
			# INCREMENT HORIZONTAL GRID COUNT
	
			cell_count += 1						
			# INCREMENT TOTAL GRID COUNT

		# END


		if (im_width - (hor_cell_count * grid_width)) > 0:
			ax = hor_cell_count * grid_width
			bx = im_width
			cx = ax
			dx = bx
			# CALCULATION OF ay, by, cy, dy ARE NOT NEEDED, VALUES ARE BORROWED		

			grid_coordinates.append([ax, ay, bx, cy])
			# CALCULATION OF by, cx, dx, AND dy IS NOT NECESSARY
	
			cell_count += 1						
			# INCREMENT TOTAL GRID COUNT

		# END
	
		hor_cell_count = 0						
		# RESET HORIZONTAL GRID COUNT	
	
		ver_cell_count += 1						
		# INCREMENT VERTICAL GRID COUNT
	
	grid_coordinates = tuple(grid_coordinates)
	
	return (cell_count, grid_coordinates)
# ---------------------------------------------------------------------------------------


