import numpy as np
import pdb


class Explitcitness():
    def __init__(self, mode='baseline'):
        self.curves = []
        self.curves_names = []
        self.baseline_losses = []

        self.mode = mode
    def add_curve(self, x,y, baseline_loss=1.0, name='curve'):
        x = np.array(x)
        y = np.array(y)
        # we use the minimum loss achieved until that capacity
        for i in range(y.shape[0]):
            y[i] = y[:(i+1)].min()

        curve = np.vstack((x,y))
        self.curves.append(curve)
        self.baseline_losses.append(baseline_loss)
        self.curves_names.append(name)
    
    def get_explitcitness(self, debug=False):
        if len(self.curves) == 0:
            return {}
        # pdb.set_trace()
        # self.curves = np.array(self.curves)
        max_x = np.array([c[0].max() for c in self.curves]).max()
        min_x = np.array([c[0].min() for c in self.curves]).min()

        max_y = np.array([c[1].max() for c in self.curves]).max()
        min_y = np.array([c[1].min() for c in self.curves]).min()

        for ind_c, curve in enumerate(self.curves):
            new_x = list(curve[0])
            new_y = list(curve[1])
            # we add a virtual point with maximum capacity but same performance as the previous one
            # we assume we have reached convergence
            # TODO: alternatively we can do the oposite, remove all above a maximum capacity
            if curve[0,-1] < max_x:
                new_x.append(max_x)
                new_y.append(new_y[-1])
        
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            new_curve = np.vstack((new_x,new_y))
            self.curves[ind_c] = np.array(new_curve)
        # max_x = self.curves[:,0,:].max()
        # min_x = self.curves[:,0,:].min()
        
        # max_y = self.curves[:,1,:].max()
        # min_y = self.curves[:,1,:].min()
        max_area = (max_x - min_x) * (max_y - min_y) # this has a disadvantage. it depends on whole group of curves. 
        max_area2 = (max_x - min_x) * 1.0 

        # TODO: add another point to each curve that represents a constant line between 
        # c_max of the curve and the global c_MAX of any curve
        all_E = {}
        for i, name in enumerate(self.curves_names):
            # this has the small advantage over max_area2 that for classes with small baseline the differences in E are bigger
            max_area3 = (max_x - min_x) * self.baseline_losses[i] 

            if self.mode == 'global':
                E = compute_explitcitness(self.curves[i][0], self.curves[i][1], 
                    global_max_area = max_area3,
                    baseline_loss=self.baseline_losses[i],
                    name=name,
                    debug=debug
                    )
            else:
                E = compute_explitcitness(self.curves[i][0], self.curves[i][1], 
                    # global_max_area = max_area,
                    max_x = max_x,
                    min_x = min_x,
                    baseline_loss=self.baseline_losses[i],
                    name=name
                    )
            all_E[name] = E
        return all_E


def compute_explitcitness(x,y, global_max_area=None, max_x = None, min_x=None, baseline_loss = 1.0, name='',
        debug=False):
    x = np.array(x)
    y = np.array(y)

    min_y_index = np.argmin(y)
    min_y = y[min_y_index]

    dy = 0.5 * (y[:-1] - min_y) + 0.5 * (y[1:] - min_y)  
    dx = x[1:] - x[:-1]

    area_under = dy * dx
    # if max_area is not None:
    #     # max_area = (y.max() - y.min()) * (x.max() - x.min())
    print(f'[{name}] use baseline: {baseline_loss}')
    # else:
    if (global_max_area is None) and (max_x is not None) and (min_x is not None):
        max_area = (baseline_loss - y.min()) * (max_x - min_x)
    else:
        max_area = global_max_area
    if debug:
        print(f'name {name}: {area_under.sum()} / {max_area} = {area_under.sum() / max_area}')
        print(f'name {name}: dy: {dy}')
        print(f'name {name}: {area_under.sum() / max_area}')
        # pdb.set_trace()
    E = 2 * area_under / max_area
    E = E.sum()
    E = 1.0 - E
    return E

if __name__ == "__main__":
    x = [1,2,3,4,5,6,7]
    y = np.array([5,4,3,2,1,0,0]) / 6
    # E = 1

    # E = compute_explitcitness(x,y)
    # print(E)

    E_metric = Explitcitness()
    E_metric.add_curve(x,y,'linear_1')

    x2 = [1,2,3,4,5,6]
    y2 = np.array([2,2,2,2,2,2]) / 2.2
    E_metric.add_curve(x2,y2,'constant')


    x3 = [1,2,3,4,5,6]
    y3 = np.array([4,3,3,2,2,2]) / 5
    E_metric.add_curve(x3,y3,'linear_slower')

    x4 = [1,2,3,4,5]
    y4 = np.array([4,3,3,2.5,2.5]) / 4
    E_metric.add_curve(x4,y4,'linear_slower2')

    x4 = [1,2,3,4,5]
    y4 = np.array([4,3,3,2.5,10]) / 4
    E_metric.add_curve(x4,y4,'linear_slower_overfit')


    all_E = E_metric.get_explitcitness()
    [print(f'{name} E: {val}') for name, val in all_E.items()]
