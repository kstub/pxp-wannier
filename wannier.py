import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def project_pos_onto_basis(vecs, pos):
    """
    Given a basis for an orthogonal projector (B) and a position operator (X),
    return the non-zero eigenvalues and eigenvectors of PXP where P = B B^*.

    Input:
    vecs -- A N x M matrix with orthogonal columns. The columns of vecs span the
            range of the projector P.
    pos -- A N x N self-adjoint position matrix.
    """
    tol = 1e-5
    rnk = vecs.shape[1]

    E, V_sm = LA.eigh(vecs.conj().T @ pos @ vecs)

    V = vecs @ V_sm
        
    return (E,V)

def sub2ind(x, b):
    """
    Converts an ordered tuple of indexes to a linear index.
    """
    return np.ravel_multi_index(x,b,mode='wrap')


class HaldaneModel:
    def __init__(self, **kw):
        req_params = [ "N", "M", "t", "tp", "v", "phi", "bdy_cond" ]

        for k in req_params:
            if k in kw.keys():
                self.__dict__[k] = kw[k]
            else:
                raise ValueError("Missing parameter: '%s'" % ( k ))

        if "noise" not in kw.keys():
            self.noise = 0
        else:
            self.noise = kw["noise"]
            
        self.B = None
        self.P = None
        self.PXP_V = None
        self.W = None

    def diag_PXP(self, pos):
        """
        First, diagonalizes the Hamiltonian for this system to calculate a basis
        for the Fermi projection. Then uses this basis for the Fermi projection
        to diagonalize PXP. Returns the eigenvalues of PXP and the gap of H. The
        basis for the Fermi projection is stored in self.B and the eigenvectors
        of PXP are stored in self.PXP_V.

        Input:
        pos -- A N x N self-adjoint position matrix. Must be the same size as
               the system.
        """
        num_cells = self.N * self.M

        self.H = self.gen_H(self.noise)
        (E_H, V_H) = np.linalg.eigh(self.H)

        print("Gap: %f" % (E_H[num_cells] - E_H[num_cells-1]))
        print(E_H[num_cells-3:num_cells+3])

        self.B = V_H[:,:num_cells]
        self.P = self.B @ self.B.conj().T


        PXP_E, self.PXP_V = project_pos_onto_basis(self.B, pos)

        return PXP_E, E_H[num_cells] - E_H[num_cells-1]
    

    def diag_PjYPj(self, pos, bins, j0=None):
        """
        Given a position operator (Y) and a selection of bins, find the
        eigenfunctions of PjYPj. This functions uses self.PXP_V to define
        PjYPj. The results of this function are stored in self.W.

        Input:
        pos -- An N x N self-adjoint position matrix. Must be same as system
                size.
        bins -- A list of 'bins' to define the projectors Pj. An example bins
                list would be [0, 50, 100]. This would correspond to setting P0
                to be the projector onto the first 50 eigenvectors of PXP
                (=range(0, 50)) and setting P1 to be the projector onto the last
                50 eigenvectors (=range(50, 100)) of PXP.
        j0 (optional) -- Can specify which j to use when diagonalizing
                 PjYPj. Default is None which corresponds to diagonalizing PjYPj
                 for all j
        """
        if self.PXP_V is None:
            raise ValueError("Error. Must diagonalize PXP first")
        
        if self.W is None:
            self.W = np.zeros(self.PXP_V.shape, dtype=complex)
                
        if j0 is not None:
            Vj_range = range(bins[j0], bins[j0+1])
            Vj = self.PXP_V[:,Vj_range]

            _, Wj = project_pos_onto_basis(Vj, pos)
            self.W[:,Vj_range] = Wj

            return Wj
        else:
            for j0 in range(len(bins)-1):
                Vj_range = range(bins[j0], bins[j0+1])
                Vj = self.PXP_V[:,Vj_range]
                print(Vj_range)

                _, Wj = project_pos_onto_basis(Vj, pos)
                self.W[:,Vj_range] = Wj

    @staticmethod
    def gen_position_ops(N, M):
        """
        Returns a tuple of the standard position operators. These operators are
        (2NM) x (2NM) self-adjoint matrices, where N and M are the supplied
        inputs.
        """
        num_cells = N * M

        X = [ None ] * (2 * num_cells)
        Y = [ None ] * (2 * num_cells)
        for (i_N, i_M) in np.ndindex(N, M):
            idx = sub2ind((i_N, i_M), (N, M))

            X[idx] = i_N
            X[idx + num_cells] = i_N

            Y[idx] = i_M + 1
            Y[idx + num_cells] = i_M 

        return np.diag(X), np.diag(Y)

    def expected_bins(self, pos_name):
        """
        Returns the expected bins for this system. Since this Haldane model is
        on a rectangular lattice, this corresponds to amounts to keeping track
        of how degenerate X and Y are.
        """
        if pos_name == "X":
            return np.array(range(0, self.N * self.M + 1, self.M))
        elif pos_name == "Y":
            return np.array(range(0, self.N * self.M + 1, self.N))

    def gen_H(self, noise=0):
        """
        Returns a Haldane Hamiltonian with the supplied parameters.
        """
        if (self.bdy_cond is not "periodic") and (self.bdy_cond is not "dirichlet"):
            raise ValueError("Error. Invalid self.bdy_cond: %s. Only 'periodic' and 'dirichlet' are supported" % ( self.bdy_cond ))

        num_cells = self.N * self.M
        phase = np.exp(1j * self.phi)
        is_in_bounds = lambda x : (0 <= x[0] and x[0] < self.N) and (0 <= x[1] and x[1] < self.M)

        
        
        H = np.zeros((2 * num_cells, 2 * num_cells), dtype=complex)
        H += np.diag([self.v] * num_cells + [-self.v] * num_cells)
        if noise > 0:
            eta = noise * np.diag([ i for i in np.random.randn(2 * num_cells) ])
            H += eta
        
        for (i_N, i_M) in np.ndindex((self.N, self.M)):
            idx = sub2ind((i_N, i_M), (self.N, self.M))

            # Nearest Neighbors
            A_neighbor_list = ((i_N - 1, i_M), (i_N, i_M - 1), (i_N, i_M))
            B_neighbor_list = ((i_N + 1, i_M), (i_N, i_M + 1), (i_N, i_M))
        
            # (Next) Nearest Neighbors
            AB_nneighbor_list1 = ((i_N, i_M - 1), (i_N + 1, i_M), (i_N - 1, i_M + 1))
            AB_nneighbor_list2 = ((i_N, i_M + 1), (i_N - 1, i_M), (i_N + 1, i_M - 1))
                
            for x in A_neighbor_list:
                if self.bdy_cond is "dirichlet" and (not is_in_bounds(x)):
                    continue
                H[idx, num_cells + sub2ind(x, (self.N, self.M))] = self.t
            
            for x in B_neighbor_list:
                if self.bdy_cond is "dirichlet" and (not is_in_bounds(x)):
                    continue
                H[num_cells + idx, sub2ind(x, (self.N, self.M))] = self.t

            for x in AB_nneighbor_list1:
                if (self.bdy_cond is "dirichlet") and (not is_in_bounds(x)):
                    continue            
                H[idx, sub2ind(x, (self.N, self.M))] = self.tp * phase.conj()
                H[num_cells + idx, num_cells + sub2ind(x, (self.N, self.M))] = self.tp * phase

            for x in AB_nneighbor_list2:
                if (self.bdy_cond is "dirichlet") and (not is_in_bounds(x)):
                    continue
                H[idx, sub2ind(x, (self.N, self.M))] = self.tp * phase
                H[num_cells + idx, num_cells + sub2ind(x, (self.N, self.M))] = self.tp * phase.conj()

        return H


    def plot_function(self, v, plot_type, title=None, filename=None,
    pad_value=None):
        """
        Generates a 2D or 3D pyplot figure with the supplied vector v.

        Inputs:
        v -- A vector representating a state for the Haldane model
        plot_type -- A string specifying how to plot the vector v. Currently
                     there are four different modes:
                        "2d" : Generates a 2d plot (log scale) in a pyplot figure
                        "3d" : Generates a 3d plot in a pyplot figure
                        "both" : Generates both a 2d plot (log scale) and 3d
                                 plot 
                        "data" : Returns a version of v reshaped to 2D, no
                                 pyplot figures are genreated (used to for
                                 custom plots)
        title -- Sets the title of the plot. If None, no title is used.
        filename -- The location of where to save the pyplot figure after it is
                    generated. If None then the plot is not saved.
        pad_value -- Used to pad the vector v with zeros when using Dirichlet
                     boundary conditions.
        """
        dims = (self.N, self.M)
        AtB_step = np.prod(dims)
        plot_array = np.zeros(dims)

        for (i1, i2) in np.ndindex(dims[0], dims[1]):
            idx = sub2ind((i1, i2), dims)
            tmp_v = (v[idx], v[idx + AtB_step])
            plot_array[i1, i2] = np.linalg.norm(tmp_v)
        plot_array = plot_array.T

        def draw_2d_plot(ax1, fig1, to_plot):
            img = ax1.imshow(to_plot[::-1,:], norm=LogNorm(), extent=[0, dims[0], 0, dims[1]])
            ax1.set_xlabel(r"$X$", fontsize=15)
            ax1.set_ylabel(r"$Y$", fontsize=15)

            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.2)
            fig1.colorbar(img, cax=cax1)

        def draw_3d_plot(ax1, fig1, to_plot, pad_value):
            nn, mm = np.meshgrid(range(-1, dims[1]+1), range(-1, dims[0]+1))
            padded = np.pad(to_plot, [(1, 1), (1, 1)], mode='constant', constant_values=(pad_value, pad_value))
            
            ax1.view_init(25, 240)
            ax1.plot_surface(nn, mm, padded[::,:])
            ax1.set_xlabel(r"$X$", fontsize=15)
            ax1.set_ylabel(r"$Y$", fontsize=15)  

            
        if plot_type == "data":
            return plot_array
        
        elif plot_type == "2d":
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(1, 1, 1)
            draw_2d_plot(ax, fig, plot_array)            
                        
        elif plot_type == "3d":
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111, projection='3d')
            draw_3d_plot(ax, fig, plot_array, pad_value)
            
        elif plot_type == "both":
            fig = plt.figure(figsize=(5, 12))
            ax0 = fig.add_subplot(2, 1, 1, projection='3d')
            ax1 = fig.add_subplot(2, 1, 2)

            draw_3d_plot(ax0, fig, plot_array, pad_value)
            draw_2d_plot(ax1, fig, plot_array)
            
        else:
            raise ValueError("Plot type '%s' invalid." % ( plot_type ))


        if title is not None:
            fig.suptitle(title)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + "_" + plot_type + ".png", bbox_inches='tight')
            plt.show()
            
