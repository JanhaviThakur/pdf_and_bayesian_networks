"""
Created on Fri Sept 13 15:52:40 2018

@author: JanhaVi
"""


import numpy as np
import xlrd as xl
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath
from scipy.stats import multivariate_normal,norm

Location = join(dirname(abspath(__file__)), 'data', 'university data.xlsx');
workbook = xl.open_workbook(Location, encoding_override="cp1252")
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[0])
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def plot_pairwise_graph(y_axis_left, y_axis_right, graph_title, x_title, y_title_left, y_title_right):
   sequence_for_unis = np.arange(1,50)
   fig, ax1 = plt.subplots()
   ax1.plot(sequence_for_unis,y_axis_left ,'b')
   ax1.set_xlabel('Universities')
   # Make the y-axis label and tick labels match the line color.
   for t in ax1.get_yticklabels():
    t.set_color('b')    
   ax1.set_ylabel(y_title_left, color='b')   
   
   ax2 = ax1.twinx()
   ax2.plot(sequence_for_unis, y_axis_right, 'r')
   for t in ax2.get_yticklabels():
    t.set_color('r')
   ax2.set_ylabel(y_title_right, color='r')
   plt.title(graph_title)
   plt.autoscale(tight=True)
   return 

# print name and person number
print('UBit name= jthakur');
print('person no= 50206922');
CS_score = [[sheet.cell_value(r, 2) ]for r in range(sheet.nrows)];
CS_score=CS_score[1:50];
CS_score=np.asarray(CS_score);
plt.plot(CS_score,'b*');plt.title('cs score graph');plt.xlabel('Universities');plt.ylabel('cs score');

Res_Over= [[sheet.cell_value(r, 3) ]for r in range(sheet.nrows)];
Res_Over=Res_Over[1:50];
Res_Over=np.asarray(Res_Over);
plt.figure(2);
plt.plot(Res_Over,'r*');plt.title('research overhead graph');plt.xlabel('Universities');plt.ylabel('research overhead ');

Ad_base = [[sheet.cell_value(r, 4) ]for r in range(sheet.nrows)];
Ad_base=Ad_base[1:50];
Ad_base=np.asarray(Ad_base);
plt.figure(3);
plt.plot(Ad_base,'g*');plt.title('Admin Base Pay $ graph');plt.xlabel('Universities');plt.ylabel('Admin Base Pay $');

Tuition = [[sheet.cell_value(r, 5) ]for r in range(sheet.nrows)];
Tuition=Tuition[1:50];
Tuition=np.asarray(Tuition);
plt.figure(4);
plt.plot(Tuition,'y*');plt.title('Tuition out of state $ graph');plt.xlabel('Universities');plt.ylabel('Tuition out of state $');

#Question 1
mu1=np.mean(CS_score);mu2=np.mean(Res_Over);mu3=np.mean(Ad_base);mu4=np.mean(Tuition);
var1=np.var(CS_score);var2=np.var(Res_Over);var3=np.var(Ad_base);var4=np.var(Tuition);
sigma1=np.std(CS_score);sigma2=np.std(Res_Over);sigma3=np.std(Ad_base);sigma4=np.std(Tuition);
print('mu1=%.3f'%mu1,'\nmu2=%.3f'%mu2,'\nmu3=%.3f'%mu3,'\nmu4=%.3f'%mu4,'\nvar1=%.3f'%var1,'\nvar2=%.3f'%var2,'\nvar3=%.3f'%var3,'\nvar4=%.3f'%var4,'\nsigma1=%.3f'%sigma1,'\nsigma2=%.3f'%sigma2,'\nsigma3=%.3f'%sigma3,'\nsigma4=%.3f'%sigma4);

# Question 2
Y=np.hstack((CS_score,Res_Over,Ad_base,Tuition));
X=np.transpose(Y);
covarianceMat=np.cov(X);
print('CovarianceMat=',covarianceMat);
correlationMat=np.corrcoef(X);
print('CorrelationMat=',correlationMat);
plot_pairwise_graph(CS_score,Res_Over,'','Universities','cs score','research overhead');
plot_pairwise_graph(CS_score,Ad_base,'','Universities','cs score','admin base pay $');
plot_pairwise_graph(CS_score,Tuition,'','Universities','cs score','tuition out-of-state $');
plot_pairwise_graph(Res_Over,Ad_base,'','Universities','research overhead','admin base pay $');
plot_pairwise_graph(Res_Over,Tuition,'','Universities','research overhead','tuition out-of-state $');
plot_pairwise_graph(Ad_base,Tuition,'','Universities','admin base pay $','tuition out-of-state $');

# Question 3
pdf1=norm.pdf(Y[:,0],mu1,sigma1);
l=np.linspace(mu1-3*sigma1,mu1+3*sigma1,100);L=norm.pdf(l,mu1,sigma1);
plt.figure();plt.plot(CS_score,pdf1,'bx',label='pdf of CSscore');plt.plot(l,L,'k-',label='pdf with mu1 and var1');plt.xlim(mu1-3*sigma1,mu1+3*sigma1);
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

pdf2=norm.pdf(Y[:,1],mu2,sigma2);
l=np.linspace(mu2-3*sigma2,mu2+3*sigma2,100);L=norm.pdf(l,mu2,sigma2);
plt.figure();plt.plot(Res_Over,pdf2,'rx',label='pdf of ResearchOverhead');plt.plot(l,L,'k-',label='pdf with mu2 and var2');plt.xlim(mu2-3*sigma2,mu2+3*sigma2);
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

pdf3=norm.pdf(Y[:,2],mu3,sigma3);
l=np.linspace(mu3-3*sigma3,mu3+3*sigma3,100);L=norm.pdf(l,mu3,sigma3);
plt.figure();p=plt.plot(Ad_base,pdf3,'gx',label='pdf of AdminBasePay');plt.plot(l,L,'k-',label='pdf with mu3 and var3');plt.xlim(mu3-3*sigma3,mu3+3*sigma3);
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

pdf4=norm.pdf(Y[:,3],mu4,sigma4);
l=np.linspace(mu4-3*sigma4,mu4+3*sigma4,100);L=norm.pdf(l,mu4,sigma4);
plt.figure();p=plt.plot(Tuition,pdf4,'yx',label='pdf of Tuition');plt.plot(l,L,'k-',label='pdf  with mu4 and var4');plt.xlim(mu4-3*sigma4,mu4+3*sigma4);
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

PdfY=pdf1*pdf2*pdf3*pdf4;
logLikelihood=sum(np.log(PdfY));
print('logLikelihood=%.3f'%logLikelihood);

#Question 4
def conditional(A,B,mua,mub):
    covar=np.cov(A,B);
    var_ab=covar[1,0];var_aa=covar[0,0];var_bb=covar[1,1];
    u_var=var_bb-(var_ab**2/var_aa);
    u_pdf=np.zeros((np.size(A),1));
    for i in range(np.size(A)):
        u_mean=mub+var_ab*(A[0,i]-mua)/var_aa;
        u_pdf[i]=norm.pdf(B[0,i],u_mean,u_var);
    return(u_pdf);

X1=np.transpose(CS_score); X2=np.transpose(Res_Over);X3=np.transpose(Ad_base);X4=np.transpose(Tuition);
px3x4=conditional(X4,X3,mu4,mu3);
px4x1=conditional(X1,X4,mu1,mu4);
px1x2=conditional(X2,X1,mu2,mu1);
px2=np.reshape(pdf2,(49,1));
pX=px3x4*px4x1*px1x2*px2;     #using markov's chain theorem
BNlogLikelihood=sum(np.log(pX)); 
BNgraph=np.array([(0, 0, 0, 1),(1, 0, 0, 0),(0, 0, 0, 0),(0, 0, 1, 0)]);
print('BNgraph=',BNgraph);
print('BNlogLikelihood=%.3f'%BNlogLikelihood);

# Question 4:(multivariate pdf)
mu=np.hstack((mu1,mu2,mu3,mu4));
P=multivariate_normal(mean=mu,cov=covarianceMat,allow_singular=True);
Pr=P.pdf(Y);
GlogLikelihood=sum(np.log(Pr));
print('logLikelihood using gaussian multivariate normal distribution=%.3f'%GlogLikelihood);
