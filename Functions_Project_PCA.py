# import numpy
import numpy as np

# NOTE: run this .py file before running Table1 and Table 2 files 

##### function 1: standardizer() #####
# standardize function to rescale the data to have mean = 0 and sd = 1

def standardizer(data):
    d = data
    means = sum(data)/data.shape[0]
    
    sds = np.sqrt((sum((data-(sum(data)/ data.shape[0]))**2))/(data.shape[0]))
    #return standardized data ((x_i - mu)/ sigma)
    return (d - means)/(sds) 
 
 
##### function 2: SSE() ##### 
 
# SSE function: calculate Sum Squared Error of Reconstruciton error 

def SSE (dat1, dat2):
   return sum(sum(((dat1) - (dat2))**2))


##### function 3: compute_u() #####

# funciton to extract single component
# Since the data has to be passed multiple times to this funciton, the data is standardized within the function rather than outside the function 

def compute_u(dat, iter, lam):

        x = standardizer(dat)

        def descent (x, u, ind1, ind2):
                id = np.ones(x.shape[1])*-1
                # implementation of the gradient descent of reconstruction error formula 
                id[ind1] = 2
                return 2*np.dot(u*id, x[ind2]) * (x[ind2,ind1] - u[ind1]*(np.dot(u, x[ind2]))) + 2*lam*u[ind1]
                 
        # starting value of u and learning rate
        u = np.ones(dat.shape[1])
        u_prev = np.zeros(dat.shape[1])
        epsilon = .1/dat.shape[0]

        # find components
        for j in range(iter+1):
                        for n in range(dat.shape[0]):
                                for i in range(dat.shape[1]):
                                        u_prev[i] = u[i]
                                        u[i] = u[i] - descent(x, u, i, n)*epsilon                         
                                  
        
        Us = np.zeros((x.shape[0], x.shape[1])) + u 
        # reconstructed data
        rec_dat = np.dot(x,u)*Us.T
        # difference between input data and reconstructed data
        dif_dat = x.T - rec_dat
        # calculate SSE for x_original - x_reconstructed
        # For now, useful only for the first time
        #SSE = sum(sum((SSE_dat)**2))
             
        return u, rec_dat, dif_dat
     
    


##### function 4: optimal_component() #####
  
def optimal_component(dat, max_iter = 50, max_comp = 10, SSE_ratio = .05, lam = 0):
   
   # give an arbitrary SSE alue to start loop
   # starting data
   x = dat  
   x_org = standardizer(x)
   rec_data_tot = np.zeros((x.shape[1],x.shape[0]))
   components = np.array([])
   SSEs = np.array([])
  
   for i in range(max_comp):
       
       u1, rec_data, new_x = compute_u(x, max_iter, lam) 
       rec_data_tot =  rec_data_tot + rec_data
       new_SSE = SSE(x_org.T, rec_data_tot)
       prev_SSE = SSE(x_org.T, rec_data_tot - rec_data)

       if (new_SSE/prev_SSE) < 1 - SSE_ratio:

         SSEs = np.append(SSEs, (new_SSE/prev_SSE))
         components = np.append(components, u1)
         x = new_x.T
         if i == max_comp-1:
          components = components.reshape(i+1, len(u1))
       else:
          components = components.reshape(i, len(u1))
          x_org = x
          break
       
   return components, (1 - SSEs)     



##### function 5: comp_features() #####

# function to create weighted features based on components (used for prediction)

def comp_feature(org_data, components, features):

  x = org_data
  u = components
  comp_features = []
  
  for i in range(features):
    # append to u
    comp_features = np.append(comp_features, np.dot(x, u[i]))
    # calculate remaining data
    Us = np.zeros((x.shape[0], x.shape[1])) + u[i]
    rec_dat = np.dot(x,u[i])*Us.T
    dif_dat = x.T - rec_dat
    x = dif_dat.T
  
  return comp_features.reshape(features,x.shape[0]).T





