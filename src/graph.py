import numpy as np
# edge_potential 4x2


class GraphChecker:
    def __init__(self,edge_potential):
        self.triangle_potential = np.ones((8,8))
        ep = np.array(edge_potential)
        for row in range(8):
            for col in range(8):
                row_i = [int(row&4!=0),int(row&2!=0),int(row&1!=0)]
                col_i = [int(col&4!=0),int(col&2!=0),int(col&1!=0)]
                p     =  1
                p    *= ep[row_i[0]*2+row_i[1],col_i[0]]
                p    *= ep[row_i[1]*2+row_i[2],col_i[1]]
                p    *= ep[row_i[0]*2+row_i[2],col_i[2]]
                self.triangle_potential[row,col] = p
    def find_inliers(self,feature3d,feature2d,triangle_ids):
        print('by graph')
        probs =[]
        for i in range(0,feature3d.shape[0]):
            probs.append([])
        for triangle_id in triangle_ids:
            data=[]
            depths   = feature3d[triangle_id,2]
            pixel_vs = feature2d[triangle_id,1]
            pa,pb,pc = get_probability(pixel_vs,depths,self.triangle_potential)
            probs[triangle_id[0]].append(pa)
            probs[triangle_id[1]].append(pb)
            probs[triangle_id[2]].append(pc)
        
        valid_id =[]
        for i in range(0,feature3d.shape[0]):
            p =get_assemble_probability(np.array(probs[i]))
            valid_id.append(p>0.5)
        return np.array(valid_id)


class GraphGrow:
    def __init__(self,threshold_angle=8):
        self.threshold_angle = threshold_angle
        self.threshold_height= 0.2
        self.graph = []
        self.proposal=[]
        self.height_invs =[]
        self.angles  =[]
    def graph_construction(self,triangle_ids):
        graph  = []
        graph_meta = []
        max_id = np.max(triangle_ids)
        for i in range(0,max_id+1):
            graph_meta.append(set())
        for i in range(0,triangle_ids.shape[0]):
            graph.append([])
        for i in range(0,triangle_ids.shape[0]):
            a,b,c = triangle_ids[i]
            intersect = graph_meta[a].intersection(graph_meta[b])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            intersect = graph_meta[a].intersection(graph_meta[c])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            intersect = graph_meta[b].intersection(graph_meta[c])
            if len(intersect)!=0:
                graph[i].append(list(intersect)[0])
                graph[list(intersect)[0]].append(i)
            graph_meta[a].add(i)
            graph_meta[b].add(i)
            graph_meta[c].add(i)
        self.graph = graph
    def check(self,i,j):
        if np.abs(self.angles[i]-self.angles[j])<self.threshold_angle and np.abs(self.height_invs[i]-self.height_invs[j])< self.threshold_height:
            return True
        else:
            return False

    def expend(self,i,proposal):
        for j in self.graph[i]:
            if (j not in proposal) and self.check(i,j):
                proposal.append(j)
                self.expend(j,proposal)

    def process(self,triangle_ids,heights,angles):
        final_proposal = []
        self.graph_construction(triangle_ids)
        self.height_invs =  1/heights
        self.angles      =  angles
        flat_flag = angles<-85
        low_flag  = self.height_invs < np.median(self.height_invs[angles<-80])
        flat_id = bool2id(flat_flag&low_flag )
        self.threshold_height = 0.4*np.median(self.height_invs)
        print(self.threshold_height)
        if len(flat_id) == 0:
            return []
        for i in range(100):
            seed = np.random.choice(flat_id)
            proposal=[seed]
            self.expend(seed,proposal)
            if len(proposal) > len(final_proposal):
                final_proposal=proposal[:]
            proposal.clear()
        print(final_proposal)
        print(len(self.height_invs),len(self.angles),len(triangle_ids))
        print(self.height_invs[final_proposal],self.angles[final_proposal])
        return final_proposal


def triangle(edge_potential):
    triangle_potential = np.ones((8,8))
    ep = np.array(edge_potential)
    for row in range(8):
        for col in range(8):
            row_i = [int(row&4!=0),int(row&2!=0),int(row&1!=0)]
            col_i = [int(col&4!=0),int(col&2!=0),int(col&1!=0)]
            p     =  1
            p    *= ep[row_i[0]*2+row_i[1],col_i[0]]
            p    *= ep[row_i[1]*2+row_i[2],col_i[1]]
            p    *= ep[row_i[0]*2+row_i[2],col_i[2]]
            triangle_potential[row,col] = p
    return triangle_potential

def check_triangle(v,d):
    a = int((v[0]-v[1])*(d[0]-d[1])<0)
    b = int((v[1]-v[2])*(d[1]-d[2])<0)
    c = int((v[0]-v[2])*(d[0]-d[2])<0)
    index = a*4+b*2+c
    return index

def get_assemble_probability(probs):
    return np.sum(probs>0.6)/len(probs)

def get_probability(v,d,tp):
    range_ = np.array([0,1,2,3,4,5,6,7])
    a_ =  (range_&4)>0
    b_ =  (range_&2)>0
    c_ =  (range_&1)>0
    index = check_triangle(v,d)
    potential = tp[:,index]
    z   = np.sum(potential)
    pa  =  np.sum(potential[a_])/z
    pb  =  np.sum(potential[b_])/z
    pc  =  np.sum(potential[c_])/z
    return [pa,pb,pc]

   
def bool2id(flag):
    id_array = np.array(range(0,flag.shape[0]))
    ids      = id_array[flag]
    return ids




def main():
    edge_potential = [[3,1],[2,2],[2,2],[0,4]]
    tp = triangle(edge_potential)
    print(tp)
    v  = [0,1,2]
    d  = [2,1,1]
    index = check_triangle(v,d)
    print(tp[:,index])
    prob = get_probability(v,d,tp)
    print(prob)



if __name__ == '__main__':
    main()

