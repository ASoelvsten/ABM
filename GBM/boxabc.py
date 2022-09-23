import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")

def read_inf(lab1,lab2,i,name1,mmd,MMD,Ptheta,param,method, SSD):

    #[lam, s_median, s_mean, sstd1, sstd2, sstd3, sstd4, lam_median, lam_mean, MLE, lam_std1, lam_std2, lam_std3, lam_std4,qtr]
    Res = np.genfromtxt(name1)

    Lam_o = Res[:,i+0]
    Lam_p = Res[:,i+8]

    MMD1 = Lam_p - Lam_o
    Ptheta1 = Res[:,i+14]
    param1 = [lab1]*len(Ptheta1)
    method1 = [lab2]*len(Ptheta1)
    SSD1 = Res[:,i+11] # (Res[:,i+11]-Res[:,i+10])/2.
    mmd1 = abs(Lam_p-Lam_o)/Lam_o

    if len(MMD) == 0:
        MMD = list(MMD1)
        Ptheta = list(Ptheta1)
        method = list(method1)
        param = list(param1)
        SSD = list(SSD1)
        mmd = list(mmd1)
    else:
        MMD.extend(list(MMD1))
        Ptheta.extend(list(Ptheta1))
        method.extend(list(method1))
        param.extend(list(param1))
        SSD.extend(list(SSD1))
        mmd.extend(list(mmd1))

    return mmd, MMD, Ptheta, param, method, SSD

def read_ABC(lab1,lab2,i,j,name1,name2,mmd, MMD,Ptheta,param,method,SSD):

#   res_lam = [lam, lam_median, lam_mean, MLE, lam_std1, lam_std2, lam_std3, lam_std4]
    data = np.genfromtxt(name1)
    PTs = np.genfromtxt(name2)

    mmd1 = abs(data[:,i+2]-data[:,0+i])/data[:,0+i]
    MMD1 = data[:,i+2]-data[:,0+i]
    Ptheta1 = PTs[:,j]
    param1 = [lab1]*len(Ptheta1)
    method1 = [lab2]*len(Ptheta1)
    SSD1 = (data[:,5+i]-data[:,4+i])/2.

    if len(MMD) == 0:
        MMD = list(MMD1)
        Ptheta = list(Ptheta1)
        method = list(method1)
        param = list(param1)
        SSD = list(SSD1)
        mmd = list(mmd1)
    else:
        MMD.extend(list(MMD1))
        Ptheta.extend(list(Ptheta1))
        method.extend(list(method1))
        param.extend(list(param1))
        SSD.extend(list(SSD1))
        mmd.extend(list(mmd1))

    return mmd, MMD, Ptheta, param, method, SSD

def read_MCMC(name,mmd,MMD,Ptheta,param,method,SSD,lab1,lab2):

    # [MMD_lam,MMD_eps,MMD_div,MMD_die,mmd_lam,mmd_eps,mmd_div,mmd_die,qtr_lam,qtr_eps,qtr_div,qtr_die,lam,eps,div,die,ssd_lam,ssd_eps,ssd_div,ssd_die]
    mcmc = np.genfromtxt(name)
    for j in range(4):
        MMD.extend(list(mcmc[:,0+j]))
        Ptheta.extend(list(mcmc[:,8+j]))
        param1 = [lab1[j]]*len(mcmc[:,0])
        method.extend([lab2]*len(mcmc[:,0]))
        param.extend(list(param1))
        SSD.extend(list(mcmc[:,16+j]))
        mmd.extend(list(mcmc[:,4+j]))

    return mmd, MMD, Ptheta, param, method, SSD

lab1 = ["$\lambda_\mathrm{C}$",r"$\epsilon$",r"$P_\mathrm{div}$",r"$P_\mathrm{cd}$"]

lab2 = r"$\mathrm{Emu.\, NN+ABC, \, 10^4}$"
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[0],lab2,0,0,"./EMULATION/ABC/Res_NN_abc.txt","./EMULATION/ABC/Prob_theta_NN.txt",[],[],[],[],[],[])
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[1],lab2,8,1,"./EMULATION/ABC/Res_NN_abc.txt","./EMULATION/ABC/Prob_theta_NN.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[2],lab2,16,2,"./EMULATION/ABC/Res_NN_abc.txt","./EMULATION/ABC/Prob_theta_NN.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[3],lab2,24,3,"./EMULATION/ABC/Res_NN_abc.txt","./EMULATION/ABC/Prob_theta_NN.txt",mmd,MMD,Ptheta,param,method, SSD)

lab2 = r"$\mathrm{Emu. GP+ABC, \, 10^4}$"
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[0],lab2,0,0,"./EMULATION/ABC/Res_GP_abc.txt","./EMULATION/ABC/Prob_theta_GP.txt",mmd,MMD, Ptheta, param, method, SSD)
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[1],lab2,8,1,"./EMULATION/ABC/Res_GP_abc.txt","./EMULATION/ABC/Prob_theta_GP.txt",mmd,MMD,Ptheta,param,method, SSD)
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[2],lab2,16,2,"./EMULATION/ABC/Res_GP_abc.txt","./EMULATION/ABC/Prob_theta_GP.txt",mmd,MMD,Ptheta,param,method, SSD)
mmd,MMD, Ptheta, param, method, SSD = read_ABC(lab1[3],lab2,24,3,"./EMULATION/ABC/Res_GP_abc.txt","./EMULATION/ABC/Prob_theta_GP.txt",mmd,MMD,Ptheta,param,method, SSD)

lab2 = r"$\mathrm{Emu.\, NN+MCMC\,(a), \, 10^4}$"
mmd, MMD, Ptheta, param, method, SSD = read_MCMC("./EMULATION/MCMC/Res_MCMC1_emu_Neural_104_clean.txt",mmd,MMD,Ptheta,param,method,SSD,lab1,lab2)

lab2 = r"$\mathrm{Emu.\, NN+MCMC\,(b), \, 10^4}$"
mmd, MMD, Ptheta, param, method, SSD = read_MCMC("./EMULATION/MCMC/Res_MCMC2_emu_Neural_104_clean.txt",mmd,MMD,Ptheta,param,method,SSD,lab1,lab2)

lab2 = r"$\mathrm{Emu.\, GP+MCMC \,(a), \, 10^4}$"
mmd, MMD, Ptheta, param, method, SSD = read_MCMC("./EMULATION/MCMC/Res_MCMC1_emu_GP_104_clean.txt",mmd,MMD,Ptheta,param,method,SSD,lab1,lab2)

lab2 = r"$\mathrm{Emu.\, GP+MCMC \,(b), \, 10^4}$"
mmd, MMD, Ptheta, param, method, SSD = read_MCMC("./EMULATION/MCMC/Res_MCMC2_emu_GP_104_clean.txt",mmd,MMD,Ptheta,param,method,SSD,lab1,lab2)

mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[0],r"$\mathrm{Inf. \, NN, \, 10^3}$",0,"./INFERENCE/Res_infe_103_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[1],r"$\mathrm{Inf. \, NN, \, 10^3}$",15,"./INFERENCE/Res_infe_103_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[2],r"$\mathrm{Inf. \, NN, \, 10^3}$",30,"./INFERENCE/Res_infe_103_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[3],r"$\mathrm{Inf. \, NN, \, 10^3}$",45,"./INFERENCE/Res_infe_103_clean.txt",mmd,MMD,Ptheta,param,method,SSD)

mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[0],r"$\mathrm{Inf. \, NN, \, 10^5}$",0,"./INFERENCE/Res_infe_105_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[1],r"$\mathrm{Inf. \, NN, \, 10^5}$",15,"./INFERENCE/Res_infe_105_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[2],r"$\mathrm{Inf. \, NN, \, 10^5}$",30,"./INFERENCE/Res_infe_105_clean.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[3],r"$\mathrm{Inf. \, NN, \, 10^5}$",45,"./INFERENCE/Res_infe_105_clean.txt",mmd,MMD,Ptheta,param,method,SSD)

mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[0],r"$\mathrm{Inf. \, GP, \, 10^4}$",0,"./INFERENCE/Res_infe_104_clean_RBF.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[1],r"$\mathrm{Inf. \, GP, \, 10^4}$",15,"./INFERENCE/Res_infe_104_clean_RBF.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[2],r"$\mathrm{Inf. \, GP, \, 10^4}$",30,"./INFERENCE/Res_infe_104_clean_RBF.txt",mmd,MMD,Ptheta,param,method,SSD)
mmd,MMD, Ptheta, param, method, SSD = read_inf(lab1[3],r"$\mathrm{Inf. \, GP, \, 10^4}$",45,"./INFERENCE/Res_infe_104_clean_RBF.txt",mmd,MMD,Ptheta,param,method,SSD)

df = {"MMD": MMD,
      "Median": mmd,
      "Ptheta": Ptheta,
      "Param.": param,
      "Method": method,
      "SSD": SSD,
      "Par": param,
       }

df = pd.DataFrame(df)

df[df==-np.inf]=np.nan
df = df.dropna()

fsize = 14

Small = False

if Small:

    df1 = df[df["Method"].str.contains("GP")==False]
    df2 = df[df["Method"].str.contains("GP")==True]

if not Small:
    plt.figure(figsize=(30,15))

    plt.subplot(4, 1, 1)
    ax = sns.boxplot(x="Method", y="MMD", hue="Param.",
               data=df, palette="muted")

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('')
    plt.ylabel(r"$\mu_\mathrm{pred}-\theta_\mathrm{o}$",fontsize=fsize)
    plt.legend(bbox_to_anchor=(0., 1.30), ncol =7,  loc=2, borderaxespad=0.,fontsize=fsize)
    plt.yticks(size=fsize)


    plt.subplot(4, 1, 2)
    ax = sns.boxplot(x="Method", y="Median", hue="Param.",
               data=df, palette="muted")

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('')
    plt.ylabel(r"$|\mu_\mathrm{pred}-\theta_\mathrm{o}|/\theta_\mathrm{o}$",fontsize=fsize)
    plt.yscale("log")
    plt.legend([],[], frameon=False)
    plt.yticks(size=fsize)

    plt.subplot(4, 1, 3)
    ax = sns.boxplot(x="Method", y="SSD", hue="Param.",
               data=df, palette="muted")

    plt.setp(ax.get_xticklabels(), visible=False)
    ax.axes.xaxis.set_visible(False)
    plt.ylabel(r"$\sigma_\mathrm{pred}$",fontsize=fsize)
    plt.legend([],[], frameon=False)
    plt.yticks(size=fsize)

    plt.subplot(4, 1, 4)
    ax = sns.boxplot(x="Method", y="Ptheta", hue="Param.",
               data=df, palette="muted")
    plt.ylabel(r"$-\log_{10} q(\theta_\mathrm{o})$",fontsize=fsize)
    plt.xlabel(r"$\mathrm{Method}$",fontsize=fsize)
    plt.ylim(-5,10)
    plt.legend([],[], frameon=False)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)

    plt.savefig("GBM_EmuInfe_param_estimates.eps",bbox_inches='tight')

else:
    plt.figure(figsize=(20,10))

    plt.subplot(2, 1, 1)
    ax = sns.boxplot(x="Method", y="MMD", hue="Param.",
               data=df1, palette="muted")

#    plt.ylim(-5,10)
    plt.xlabel('')
    plt.ylabel(r"$\mu_\mathrm{pred}-\theta_\mathrm{o}$",fontsize=fsize)
    plt.legend(bbox_to_anchor=(0., 1.20), ncol =7,  loc=2, borderaxespad=0.,fontsize=fsize)
    plt.yticks(size=fsize)
    plt.xticks(size=fsize)

    plt.subplot(2,1,2)
    ax = sns.boxplot(x="Method", y="MMD", hue="Param.",
               data=df2, palette="muted")
    plt.ylabel(r"$\mu_\mathrm{pred}-\theta_\mathrm{o}$",fontsize=fsize)
    plt.xlabel(r"$\mathrm{Method}$",fontsize=fsize)
#    plt.ylim(-5,10)
    plt.legend([],[], frameon=False)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.savefig("GBM_EmuInfe_param_estimates_small.eps",bbox_inches='tight')

plt.show()
