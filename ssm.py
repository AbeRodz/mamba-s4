import torch 
import torch.nn as nn
import torch.nn.functional as F


class SSM(nn.Module):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

    def random_SSM(self, N: int) -> tuple[torch.Tensor]:
        """
        Generates a random state space.
        """
        A = torch.rand((N, N))
        B = torch.rand((N, 1))
        C = torch.rand((1, N))
        return A, B, C
        
    def discretize(self, A : torch.Tensor, B: torch.Tensor , C: torch.Tensor , step : float):
        """
        Discretization via bilinear transform.
        """
        identity = torch.eye(*A.size(), out = torch.empty_like(A))
        
        common = torch.linalg.inv(identity - (step/2) * A)

        _A = torch.matmul(common  ,(identity + (step/2)*A))

        _B = torch.matmul(common,  (step*B))

        return _A, _B , C
    
    def discretize_zoh(self, A : torch.Tensor, B : torch.Tensor, step : float):
        common = (step*A)
        identity = torch.eye(*A.size(), out = torch.empty_like(A))
        
        A_bar = torch.exp(common)
        B_bar = torch.linalg.inv(common)@torch.exp(common - identity ) @ (step*B)
        return A_bar, B_bar
    
    @staticmethod
    def __complex_log(float_input, eps=1e-6):
        eps = float_input.new_tensor(eps)
        real = float_input.abs().maximum(eps).log()
        imag = (float_input < 0).to(float_input.dtype) * torch.pi
        return torch.complex(real, imag)
    
    def parallel_scan(self, h_t: torch.Tensor ,A_bar : torch.Tensor, B_bar: torch.Tensor, C: torch.Tensor):
        
        # h_t = A_bar * h_t-1 + B_bar*x_t
        #y_t =C*h_t

        ## heinsen
        # x_t = a_t*x_t-1 + b_t

        # equivalent notation

        # h_t = A_bar_t * h_t-1 + B_bar*x_t
        # where x_t is the input sequence hence:
        # cat(B_bar,  x_t) = b_t' or b_t prime.
        # finally:
        # h_t = A_bar_t * h_t-1 + b_t'
        """
        In summary we need to calculate a state space equation (S4) using the heinsen method.

        """
        #b_t_prime =  torch.cat([h_t[..., None], B_bar], dim=-1)
        #b_t_prime = torch.matmul(B_bar,h_t)
        b_t_prime = B_bar @ h_t
        log_b_t = self.__complex_log(b_t_prime)
        log_a_t = self.__complex_log(A_bar)
        a_star = F.pad(torch.cumsum(log_a_t, dim = -1),(1,0))

        log_h0_plus_b_star = torch.logcumsumexp(log_b_t - a_star, dim = -1)

        log_h = (a_star + log_h0_plus_b_star)[...,1:]
        
        h = torch.exp(log_h).real
        print(h.shape)
        #y_t = torch.matmul(C.unsqueeze(-1), h).squeeze(-1)
        y_t = torch.einsum("bnt,bdnt->bdt", C, h) 
        return h, y_t
    

if __name__ == "__main__":
    pass