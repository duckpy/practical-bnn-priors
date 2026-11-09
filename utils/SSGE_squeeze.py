import torch
import torch.autograd as autograd

"""

The original implementation can be found here https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/score_estimator/spectral_stein.py

"""


class SpectralSteinEstimator():
    def __init__(self, eta=None, num_eigs=None, K=None, xm = None, device = None):
        self.eta = eta
        self.num_eigs = num_eigs
        self.K = K
        self.xm = xm
        if xm is not None:
            self.beta, self.eigen_vals, self.eigen_vec = self.compute_beta(xm)
        self.device = device 

    def nystrom_method(self, x, eval_points, eigen_vecs, eigen_vals):
        """
        Implements the Nystrom method for approximating the
        eigenfunction (generalized eigenvectors) for the kernel
        at x using the M eval_points (x_m). It is given
        by -
         .. math::
            phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)
        :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        :param eval_points: (Tensor) Sample points from the data of ize M [M x D]
        :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        :return: Eigenfunction at x [N x M]
        """
        M = torch.tensor(eval_points.size(-2), dtype=torch.float)

        Kxxm = self.K(x, eval_points)
        phi_x = torch.sqrt(M) * Kxxm @ eigen_vecs

        phi_x *= 1. / eigen_vals[:, 0]  # Take only the real part of the eigenvals
        # as the Im is 0 (Symmetric matrix)
        return phi_x

    def compute_beta(self,xm):

        M = torch.tensor(xm.size(-2), dtype=torch.float)

        xm = xm.detach().requires_grad_(True)

        Kxx = self.K(xm, xm.detach())

        dKxx_dx = autograd.grad(Kxx.sum(), xm)[0]

        # Kxx = Kxx + eta * I
        if self.eta is not None:
            Kxx += self.eta * torch.eye(xm.size(-2)).to(self.device)

        eigen_vals, eigen_vecs = eig_like_torch_eig(Kxx)

        if self.num_eigs is not None:
            eigen_vals = eigen_vals[:self.num_eigs]
            eigen_vecs = eigen_vecs[:, :self.num_eigs]



        # Compute the Monte Carlo estimate of the gradient of
        # the eigenfunction at x
        dKxx_dx_avg = -dKxx_dx/xm.shape[0] # [M x D]

        beta = - torch.sqrt(M) * eigen_vecs.t() @ dKxx_dx_avg
        beta *= (1. / eigen_vals[:, 0].unsqueeze(-1))

        return beta, eigen_vals, eigen_vecs

    def compute_score_gradients(self, x, xm = None):
        """
        Computes the Spectral Stein Gradient Estimate (SSGE) for the
        score function. The SSGE is given by
        .. math::
            \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)
            \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)
            \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)
        :param x: (Tensor) Point at which the gradient is evaluated [N x D]
        :param xm: (Tensor) Samples for the kernel [M x D]
        :return: gradient estimate [N x D]
        """
        if xm is None:
            xm = self.xm
            beta = self.beta
            eigen_vecs = self.eigen_vecs
            eigen_vals = self.eigen_vals
        else:
            beta,eigen_vals,eigen_vecs = self.compute_beta(xm)

        phi_x = self.nystrom_method(x, xm, eigen_vecs, eigen_vals)  # [N x M]
        # assert beta.allclose(beta1), f"incorrect computation {beta - beta1}"
        g = phi_x @ beta  # [N x D]
        return g
    

def eig_like_torch_eig(A, tol=1e-12):
    """
    Compute eigenpairs using torch.linalg.eig and return results matching the
    legacy torch.eig(K, eigenvectors=True) layout.

    Returns:
      eigvals_realimag: (n,2) real tensor where columns are [real, imag]
      eigvecs_packed:   (n,n) real tensor with LAPACK-style packing:
                        - real eigenvalue -> single column (real eigenvector)
                        - complex pair a+ib, a-ib -> two columns:
                          col j = real part x of eigenvector for a+ib
                          col j+1 = imag part y of eigenvector for a+ib
    Notes:
      - Pairs conjugate eigenvalues (within tol) and places the +imag one first.
      - If ordering from torch.linalg.eig already had conjugates adjacent, this still works.
    """
    if not A.is_floating_point():
        A = A.float()

    n = A.shape[0]
    # compute complex eigenpairs
    Lc, Vc = torch.linalg.eig(A)   # Lc: (n,), Vc: (n,n), complex dtype

    # Convert to CPU floats for matching & comparisons if desired, but keep device for outputs
    device = A.device
    real_dtype = A.real.dtype

    # Create arrays for bookkeeping
    used = [False] * n
    order = []  # indices in the order we'll output (real singles and paired indices in sequence)

    # We'll match conjugate pairs: for each unprocessed index i, try to find its conjugate
    for i in range(n):
        if used[i]:
            continue
        li = Lc[i]
        if abs(li.imag) <= tol:
            # real eigenvalue -> single
            used[i] = True
            order.append(i)
            continue

        # complex eigenvalue: search for conjugate j (imag opposite, real close)
        conj_found = -1
        for j in range(i+1, n):
            if used[j]:
                continue
            lj = Lc[j]
            if (abs(li.real - lj.real) <= 1e-8 * max(1.0, abs(li.real.item()), abs(lj.real.item()))
                and abs(li.imag + lj.imag) <= 1e-8 * max(1.0, abs(li.imag.item()), abs(lj.imag.item()))):
                conj_found = j
                break

        if conj_found >= 0:
            # place +imag first (ensure i is +imag, conj_found is -imag)
            if li.imag < 0:
                # swap so positive-imag comes first
                i_pos, i_neg = conj_found, i
            else:
                i_pos, i_neg = i, conj_found
            used[i_pos] = True
            used[i_neg] = True
            order.append(i_pos)
            order.append(i_neg)
        else:
            # no conjugate found later — try searching earlier indices
            conj_found2 = -1
            for j in range(0, i):
                if used[j]:
                    continue
                lj = Lc[j]
                if (abs(li.real - lj.real) <= 1e-8 * max(1.0, abs(li.real.item()), abs(lj.real.item()))
                    and abs(li.imag + lj.imag) <= 1e-8 * max(1.0, abs(li.imag.item()), abs(lj.imag.item()))):
                    conj_found2 = j
                    break
            if conj_found2 >= 0:
                # ensure positive-imag first
                if li.imag < 0:
                    i_pos, i_neg = conj_found2, i
                else:
                    i_pos, i_neg = i, conj_found2
                used[i_pos] = True
                used[i_neg] = True
                order.append(i_pos)
                order.append(i_neg)
            else:
                # fallback: no conjugate found; treat as single (this is unusual)
                used[i] = True
                order.append(i)

    # Build eigvals_realimag and eigvecs_packed according to 'order'
    eigvals_realimag = torch.zeros((n, 2), dtype=real_dtype, device=device)
    eigvecs_packed = torch.zeros((n, n), dtype=real_dtype, device=device)

    out_col = 0
    idx = 0
    while idx < len(order):
        i = order[idx]
        li = Lc[i]
        eigvals_realimag[out_col, 0] = li.real.to(real_dtype).to(device)
        eigvals_realimag[out_col, 1] = li.imag.to(real_dtype).to(device)

        if abs(li.imag) <= tol:
            # real eigenvalue -> put its real eigenvector in one column
            eigvecs_packed[:, out_col] = Vc[:, i].real.to(real_dtype).to(device)
            out_col += 1
            idx += 1
        else:
            # complex pair: order should have i (pos imag) then j (neg imag)
            if idx + 1 >= len(order):
                # defensive: no partner found (shouldn't happen)
                eigvecs_packed[:, out_col] = Vc[:, i].real.to(real_dtype).to(device)
                eigvecs_packed[:, out_col + 1] = Vc[:, i].imag.to(real_dtype).to(device)
                out_col += 2
                idx += 1
            else:
                j = order[idx + 1]
                # store eigvals for j as well
                lj = Lc[j]
                eigvals_realimag[out_col + 1, 0] = lj.real.to(real_dtype).to(device)
                eigvals_realimag[out_col + 1, 1] = lj.imag.to(real_dtype).to(device)

                # pack real and imag of the +imag eigenvector (use whichever has positive imag)
                if li.imag >= 0:
                    vpos = Vc[:, i]
                else:
                    vpos = Vc[:, j]
                eigvecs_packed[:, out_col]     = vpos.real.to(real_dtype).to(device)
                eigvecs_packed[:, out_col + 1] = vpos.imag.to(real_dtype).to(device)

                out_col += 2
                idx += 2

    # If by some ordering mismatch fewer columns were filled, that's surprising, but we return the arrays.
    return eigvals_realimag, eigvecs_packed