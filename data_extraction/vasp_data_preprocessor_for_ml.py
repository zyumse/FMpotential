import numpy as np

def generate_atom_types(natom):
    type_atom = ['Si'] * (natom // 3) + ['O'] * (natom // 3 * 2)
    return np.array(type_atom)

def extract_from_xml(folder_name, natom):
    with open(f'{folder_name}/vasprun.xml', 'r') as fxml:
        lines = fxml.readlines()

    coors_list, forces_list, e_list, box = [], [], [], np.zeros([3,3])
    for l in range(len(lines)):
        if (box[0,0] == 0) and (lines[l] == '   <varray name="basis" >\n'):
            for la in range(3):
                box[la, :] = np.array([float(it) for it in lines[l+la+1].split()[1:4]])
        elif lines[l] == '   <varray name="positions" >\n':
            coors = np.zeros([natom, 3])
            for la in range(natom):
                coors[la, :] = np.array([float(it) for it in lines[l+la+1].split()[1:4]])
            coors_list.append(coors)
        elif lines[l] == '  <varray name="forces" >\n':
            forces = np.zeros([natom, 3])
            for la in range(natom):
                forces[la, :] = np.array([float(it) for it in lines[l+la+1].split()[1:4]])
            forces_list.append(forces)
        elif '<i name="total">' in lines[l]:
            e_list.append(float(lines[l-6].split()[2]))
    
    return coors_list, forces_list, e_list, box/10

def compute_features_and_targets(type_atom, coors_list, forces_list, box, sf, ef, natom, delta_r, r_max):
    r = np.arange(delta_r, r_max, delta_r)
    X, y_force = [], []
    
    for i in range(sf, ef):
        coors = coors_list[i] @ box
        rdis = compute_relative_displacement(coors_list[i], box, natom)
        dis = compute_absolute_displacement(rdis)
        
        for idx, atom_type in enumerate(type_atom):
            if atom_type == 'Si':
                n1 = sum(type_atom == 'Si')
                other_atom = 'O'
            else:
                n1 = sum(type_atom == 'O')
                other_atom = 'Si'
            
            g_n = compute_g_vector(idx, coors, dis, atom_type, other_atom, n1, r, box)
            X.append(np.concatenate(g_n, axis=1))
            
        y_force.extend(forces_list[i])
        
    return np.array(X), np.array(y_force)

def compute_relative_displacement(rcoors, box, natom):
    rdis = np.zeros([natom, natom, 3])
    for i in range(natom):
        rdis[i, :, :] = rcoors[i] - rcoors
    rdis[rdis < -0.5] += 1
    rdis[rdis > 0.5] -= 1
    return rdis @ box

def compute_absolute_displacement(rdis):
    return np.sqrt(np.sum(np.square(rdis), axis=2)) * 10

def compute_g_vector(idx, coors, dis, atom_type, other_atom, n1, r, box):
    g_n = [np.zeros([3, len(r)-1]), np.zeros([3, len(r)-1])]

    for idr in range(len(r)-1):
        mask = (dis[idx, :n1] >= r[idr]) & (dis[idx, :n1] < r[idr+1])
        g_n[0][:, idr] = compute_vu(mask, idx, coors, box, r, idr)
        
        mask = (dis[idx, n1:] >= r[idr]) & (dis[idx, n1:] < r[idr+1])
        g_n[1][:, idr] = compute_vu(mask, idx, coors, box, r, idr)
        
    return g_n

def compute_vu(mask, idx, coors, box, r, idr):
    id_tmp = np.argwhere(mask)
    v_tmp = coors[id_tmp, :] - coors[idx, :]
    v_tmp[v_tmp > box[0,0]/2] -= box[0,0]
    v_tmp[v_tmp < -box[0,0]/2] += box[0,0]
    vn = 20 * v_tmp / (r[idr] + r[idr+1])
    return -np.sum(vn, axis=0)

def collect_inputs(folder_name, save_name, natom, sf=0, ef=1, delta_r=0.02, r_max=10):
    type_atom = generate_atom_types(natom)
    coors_list, forces_list, e_list, box = extract_from_xml(folder_name,natom)

    X1, y_force1 = compute_features_and_targets(type_atom, coors_list, forces_list, box, sf, ef, natom, delta_r, r_max)
    X2, y_force2 = compute_features_and_targets(type_atom, coors_list, forces_list, box, sf, ef, natom, delta_r, r_max+2)

    np.savez(f'{folder_name}/{save_name}', X1=X1, y_force1=y_force1, X2=X2, y_force2=y_force2)
