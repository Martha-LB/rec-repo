
import jax
import jax.numpy as jnp
import dataclasses
import numpy as np


# For each variable, the powers range from 0 to the maximum degree. 
# So the maximum number of coefficients is (Nx+1)*(Ny+1)*(Nz+1).
# And the formula is following.
def poly(x, y, z, co):
    result = 0
    for (i, j, k), coeff in co.items():
        result += coeff * (x ** i) * (y ** j) * (z ** k)
    return result



def coeffs(Nx,Ny,Nz,t):
    # Could use a dictionary to store the coefficients.
    coefficients = {}
    max_indices = [(Nx,0,0), (0,Ny,0), (0,0,Nz)]
    key = jax.random.PRNGKey(42)
    for i in max_indices:
        key, subkey = jax.random.split(key)
        coefficients[i] = jax.random.uniform(subkey,shape=(),minval=1,maxval=10)

    while len(coefficients) < t:
        key, subkey = jax.random.split(key)
        i, j, k = jax.random.randint(subkey,shape=(3,),minval=0,maxval=jnp.array([Nx+1,Ny+1,Nz+1]))
        i, j, k = int(i.item()), int(j.item()), int(k.item()) 
        if (i,j,k) not in coefficients:
            coefficients[(i,j,k)] = jax.random.uniform(subkey,shape=(),minval=1,maxval=10)
    return coefficients

def train_set(co, N):
    noise_frac = 0.25
    key = jax.random.PRNGKey(42)
    key,subkey = jax.random.split(key)
    x_values = jax.random.normal(subkey, shape=(N,))
    key,subkey = jax.random.split(key)
    y_values = jax.random.normal(subkey, shape=(N,))
    key,subkey = jax.random.split(key)
    z_values = jax.random.normal(subkey, shape=(N,))
    key,subkey = jax.random.split(key)
    outputs_pure = jnp.array([poly(x,y,z,co) for x,y,z in zip(x_values,y_values,z_values)])
    outputs_noise = jnp.array([poly(x,y,z,co)*(1+noise_frac*jax.random.normal(subkey, shape=())) for x,y,z in zip(x_values,y_values,z_values)])
    train_data = jnp.stack([x_values,y_values,z_values,outputs_noise],axis=1)
    return train_data

def loss(co, data):
    # return  jnp.sum((data[:,1] - f(data[:,0], param_w))**2)
    return jnp.log(jnp.sum((data[:,3] - poly(data[:,0],data[:,1],data[:,2],co)) ** 2))



@dataclasses.dataclass
class Flags:
    problem_a_and_b = False
    problem_c = False
    problem_d = True


if __name__ == '__main__':
    if Flags.problem_a_and_b:
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        Nx = jax.random.randint(subkey, shape=(), minval=1, maxval=6)
        key, subkey = jax.random.split(key)
        Ny = jax.random.randint(subkey, shape=(), minval=1, maxval=6)
        key, subkey = jax.random.split(key)
        Nz = jax.random.randint(subkey, shape=(), minval=1, maxval=6)
        key, subkey = jax.random.split(key)
        t = jax.random.randint(subkey, shape=(), minval=0, maxval=(Nx+1)*(Ny+1)*(Nz+1)+1)
        N = int(input("imput the N you want: "))
        co = coeffs(Nx,Ny,Nz,t)
        train_data = train_set(co,N)

        print("Nx:", Nx)
        print("Ny:", Ny)
        print("Nz:", Nz)
        print("t:", t)
        print(f"train dataset : {train_data}\n")


    if Flags.problem_c:
        # Using JAX automatic differentiation - autograd
        grad_loss = jax.grad(loss)

        Nx,Ny,Nz,t = 2,1,6,12
        N = 1000
        co = coeffs(Nx,Ny,Nz,t)
        train_data= train_set(co,N)
        num_epochs = 10
        learning_rate = 0.01
        num_points_per_batch = N // 5
        print("\n===== Running Stochastic Gradient Descent 1 =====")
        for epoch in range(num_epochs):
            # Get points for the current batch
            for i in range(0, N, num_points_per_batch):
                batch = train_data[i:i + num_points_per_batch]
                grad = grad_loss(co, batch)
                grad_dict = {}
                for (i, j, k) in co.keys():
                    grad_dict[(i, j, k)] = grad[i, j, k]
                for key in co.keys():
                    co[key] = co[key] - learning_rate * grad[key]
                # co = co - learning_rate * grad

            print(f"Epoch {epoch}: \nparam: {co}, \ngrad={grad}, loss={loss(co, train_data)}")


        Nx,Ny,Nz,t = 3,1,2,5
        co = coeffs(Nx,Ny,Nz,t)
        train_data= train_set(co,N)
        print("\n\n\n===== Running Stochastic Gradient Descent 2 =====")
        for epoch in range(num_epochs):
            # Get points for the current batch
            for i in range(0, N, num_points_per_batch):
                batch = train_data[i:i + num_points_per_batch]
                grad = grad_loss(co, batch)
                grad_dict = {}
                for (i, j, k) in co.keys():
                    grad_dict[(i, j, k)] = grad[i, j, k]
                for key in co.keys():
                    co[key] = co[key] - learning_rate * grad[key]

            print(f"Epoch {epoch}: \nparam: {co}, \ngrad={grad}, loss={loss(co, train_data)}")



    if Flags.problem_d:
        file_path = '/Users/marthalee/projects/mmd_data_secret_polyxyz.npy'
        data = np.load(file_path, allow_pickle=True)
        grad_loss = jax.grad(loss)

        num_epochs = 10
        learning_rate = 0.01
        num_points_per_batch = len(data) // 5

        Nx,Ny,Nz,t = 2,2,1,5
        co = coeffs(Nx,Ny,Nz,t)
        for epoch in range(num_epochs):
            # Get points for the current batch
            for i in range(0, len(data), num_points_per_batch):
                batch = data[i:i + num_points_per_batch]
                grad = grad_loss(co, batch)
                grad_dict = {}
                for (i, j, k) in co.keys():
                    grad_dict[(i, j, k)] = grad[i, j, k]
                for key in co.keys():
                    co[key] = co[key] - learning_rate * grad[key]
            print(f"Epoch {epoch}: \nparam: {co}, \ngrad={grad}, loss={loss(co, data)}")