# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 16:41:40 2016

@author: BWutzke143864
"""
import numpy as np
from PIL import Image

def co_occurence_matrix(label_header,label_data):
    cooccurence_matrix = np.dot(label_data.transpose(),label_data)
    cooccurrence_matrix_diagonal = np.diagonal(cooccurence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurence_matrix, cooccurrence_matrix_diagonal[:, None]))

    return cooccurence_matrix,cooccurrence_matrix_diagonal,cooccurrence_matrix_percentage

def cooccurence_shift_vert(Im):
    un,_ = np.unique(Im,return_counts=True)
    C = np.zeros((len(un),len(un)))
    for x in range(Im.shape[0]-1):
        for y in range(Im.shape[1]-1):
            C[Im[x,y],Im[x,y+1]] +=1
    #C_diag = np.diagonal(C)
    Prob = C*1/((Im.shape[0]-1)*(Im.shape[1]-1))
    return C, Prob

def cooccurence_shift_horiz(Im):
    un,_ = np.unique(Im,return_counts=True)
    C = np.zeros((len(un),len(un)))
    for x in range(Im.shape[0]-1):
        for y in range(Im.shape[1])[1:]:
            C[Im[x,y],Im[x+1,y]] +=1
    #C_diag = np.diagonal(C)
    Prob = C*1/((Im.shape[0]-1)*(Im.shape[1]-1))
    return C, Prob

def diag_cooccurence_shift_down(Im):
    un,_ = np.unique(Im,return_counts=True)
    C = np.zeros((len(un),len(un)))
    for x in range(Im.shape[0]-1):
        for y in range(Im.shape[1]-1):
            C[Im[x,y],Im[x+1,y+1]] +=1
    #C_diag = np.diagonal(C)
    Prob = C*1/((Im.shape[0]-1)*(Im.shape[1]-1))
    return C, Prob
    
def diag_cooccurence_shift_up(Im):
    un,_ = np.unique(Im,return_counts=True)
    C = np.zeros((len(un),len(un)))
    for x in range(Im.shape[0]-1):
        for y in range(Im.shape[1])[1:]:
            C[Im[x,y],Im[x+1,y-1]] +=1
    #C_diag = np.diagonal(C)
    Prob = C*1/((Im.shape[0]-1)*(Im.shape[1]-1))
    return C, Prob

def haralick(Im):
    
    horiz_cmat,horiz_cmat_percent = cooccurence_shift_horiz(Im)
    vert_cmat,vert_cmat_percent = cooccurence_shift_vert(Im)
    diag_up,diag_up_percent = diag_cooccurence_shift_up(Im) 
    diag_down,diag_down_percent = diag_cooccurence_shift_down(Im) 
    
    return horiz_cmat_percent,vert_cmat_percent,diag_up_percent,diag_down_percent
    
def haralick_stats(P):
    p_x = [sum([P[i,j] for j in range(P.shape[1])]) for i in range(P.shape[0])]
    p_y = [sum([P[i,j] for i in range(P.shape[0])]) for j in range(P.shape[1])]
    mu_x = sum([i*p_x[i] for i in range(len(p_x))])
    mu_y = sum([i*p_y[i] for i in range(len(p_y))])
    delta_x = np.sqrt(sum([p_x[i]*(i-mu_x)**2 for i in range(len(p_x))]))
    delta_y = np.sqrt(sum([p_y[i]*(i-mu_y)**2 for i in range(len(p_y))]))
    p_x_plus_y = [sum([P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1]) if i+j==k]) for k in range(2*P.shape[0])]
    p_x_minus_y = [sum([P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1]) if np.abs(i-j)==k]) for k in range(P.shape[0]-1)]
    HXY1 = -sum([P[i,j]*np.log(p_x[i]*p_y[j]) for i in range(P.shape[0]) for j in range(P.shape[1])])
    HXY2 = -sum([p_x[i]*p_y[j]*np.log(p_x[i]*p_y[j]) for i in range(P.shape[0]) for j in range(P.shape[1])])
    Q = np.zeros((P.shape[0],P.shape[1]))
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            Q[i,j] = sum([(P[i,k]*P[j,k])/(p_x[i]*p_y[j]) for k in range(P.shape[0])])
            
    HX = -sum([p_x[i]*np.log(p_x[i]) for i in range(P.shape[0])])
    HY = -sum([p_x[j]*np.log(p_x[j]) for i in range(P.shape[1])])
            
    return p_x,p_y,mu_x,mu_y,delta_x,delta_y,p_x_plus_y,p_x_minus_y,HXY1,HXY2,Q,HX,HY

def element_difference_moment(P,k = 2):
    return sum([(i-j)**k * P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1])])
    
def angular_second_moment(P):
    #Also called Uniformity or Energy
    return sum([P[i,j]**2 for i in range(P.shape[0]) for j in range(P.shape[1])])
    
def contrast(P):
    N = P.shape[0]
    quot2 = []
    for n in range(N-1):
        quot2 += n**2*sum([P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1]) if np.abs(i-j)==n])
    return sum(quot2)
    
def correlation(P,mu_x,mu_y,sig_x,sig_y):
    return None
    
def sum_of_squares_variance(P,mu):
    return sum([(i-mu)*P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1])])
    
def inverse_difference_moment(P):
    return sum([1/(1+(i-j)**2)*P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1])])

def homogeneity(P):
    return sum([P[i,j]/(1+np.abs(i-j)) for i in range(P.shape[0]) for j in range(P.shape[1])])
    
def sum_average(P):
    return None

def sum_variance(P,f):
    return None

def sum_entropy(P,f):
    return None
    
def entropy(P):
    return -sum([P[i,j]*np.log(P[i,j]) for i in range(P.shape[0]) for j in range(P.shape[1])])

def difference_variance(P):
    return None

def difference_entropy(P):
    return None

def meas_correlation_1(P,HXY,HX,HY):
    return None

def meas_correlation_2(P,HXY):
    return None

def max_correlation_coeff(P):
    return None  
    
def haralick_features(P):
    p_x,p_y,mu_x,mu_y,delta_x,delta_y,p_x_plus_y,p_x_minus_y,HXY1,HXY2,Q,HX,HY = haralick_stats(P)
    
    f1 = sum([P[i,j]**2 for i in range(P.shape[0]) for j in range(P.shape[1])])
    f2 = contrast(P)
    f3 = sum([((i-mu_x)*(i-mu_y)*P[i,j])/(delta_x*delta_y) for i in range(P.shape[0]) for j in range(P.shape[1])])
    f4 = sum([(i-np.mean(P))**2*P[i,j] for i in range(P.shape[0]) for j in range(P.shape[1])])
    f5 = inverse_difference_moment(P)
    f6 = sum([k*p_x_plus_y[k] for k in range(2*P.shape[0])])
    f7 = sum([(k-f6)**2*p_x_plus_y[k] for k in range(2*P.shape[0])])
    f8 = -sum([p_x_plus_y[k]*np.log(p_x_plus_y[k]) for k in range(2*P.shape[0])])
    f9 = entropy(P)
    f10 = sum([(k-sum([l*p_x_minus_y[l] for l in range(P.shape[0]-1)]))**2*p_x_minus_y[k] for k in range(P.shape[1]-1)])
    f11 = f8 = -sum([p_x_minus_y[k]*np.log(p_x_minus_y[k]) for k in range(P.shape[0]-1)])
    try:
        f12 = (f9 - HXY1)/(np.max(HX,HY))
    except:
        f12 = "skipped f12"
        print "skipped f12"
    f13 = np.sqrt(1 - np.exp(HXY2 - f9))
    #f14 = np.sqrt(second largest eigenvalue of Q) 
    return f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13#,f14

def norm_dist(x,mu,sig):
    return np.exp(-((x-mu)**2)/(2*sig**2))*1/(sig*np.sqrt(2*np.pi))
    
def create_prob_map(Im,A,B,C,D):
    P = np.zeros((Im.shape[0],Im.shape[1]))
    Im_var = np.var(Im)
    Im_sig = np.sqrt(Im_var)
    for i in range(Im.shape[0])[1:-1]:
        for j in range(Im.shape[1])[1:-1]:
            p_ul = C[Im[i-1,j-1],Im[i,j]]+norm_dist(Im[i-1,j-1],Im[i,j],Im_sig)
            p_u = B[Im[i,j-1],Im[i,j]]+norm_dist(Im[i,j-1],Im[i,j],Im_sig)
            p_ur = D[Im[i+1,j-1],Im[i,j]]+norm_dist(Im[i+1,j-1],Im[i,j],Im_sig)
            p_r = A[Im[i+1,j],Im[i,j]]+norm_dist(Im[i+1,j],Im[i,j],Im_sig)
            p_ll = C[Im[i+1,j+1],Im[i,j]]+norm_dist(Im[i+1,j+1],Im[i,j],Im_sig)
            p_l = B[Im[i,j+1],Im[i,j]]+norm_dist(Im[i-1,j+1],Im[i,j],Im_sig)
            p_lr = D[Im[i-1,j+1],Im[i,j]]+norm_dist(Im[i-1,j+1],Im[i,j],Im_sig)
            p_l = A[Im[i-1,j],Im[i,j]]+norm_dist(Im[i-1,j],Im[i,j],Im_sig)
            P[i,j] = p_ul+p_u+p_ur+p_r+p_ll+p_l+p_lr+p_l
    return P
     
def main():
    label_headers = 'Alice Bob Carol Dave Eve'.split(' ')
    label_data = np.random.randint(0,2,(10,5)) 
    cooccurence_matrix,cooccurrence_matrix_diagonal,cooccurrence_matrix_percentage = co_occurence_matrix(label_headers,label_data)
    print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))
    
if __name__=='__main__':
    main()