def trabecular_thickness(volume):
    dt = distance_transform_edt(volume)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def trabecular_distance(volume):
    dt = distance_transform_edt(1-volume)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])

def bone_fraction(image):  #calcul fraction osseuse
    pix_n_sum = np.sum(image==1)   #nbre de pixels noirs
    pix_b_sum = np.sum(image==2)   #nbre de pixels blancs
    bone_fraction = pix_b_sum/(pix_n_sum + pix_b_sum)   #fraction PAR IMAGE  
    return bone_fraction