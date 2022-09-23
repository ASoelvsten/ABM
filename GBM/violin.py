import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

sns.set_theme(style="whitegrid")

def appending(met,reso,i,name,MMD,AMD,WD,SSD,method,grid):

    Res = np.genfromtxt(name)

    # [s_median, s_mean, sstd1, sstd2, sstd3, sstd4, GSC_median, GSC_mean, GSC_std1, GSC_std2, GSC_std3, GSC_std4, WD]
    # GSC, GPP, GDS, Dead, Rim

    MMD1 =  (Res[:,i+7+4]-Res[:,i+1+4])/(Res[:,i+3+4]) #-Res[:,i+2+4])
    AMD1 = abs(Res[:,i+6+4]-Res[:,i+0+4])/Res[:,i+0+4]
    WD1 = Res[:,i+12+4]
    SSD1 = Res[:,i+9+4]/Res[:,i+3+4] # (Res[:,i+9+4]-Res[:,i+8+4])/(Res[:,i+3+4]-Res[:,i+2+4])
    method1 = [met]*len(AMD1)
    grid1 = [reso]*len(AMD1)

    if len(MMD) == 0:
        MMD = list(MMD1)
        AMD = list(AMD1)
        WD  = list(WD1)
        SSD = list(SSD1)
        method = list(method1)
        grid = list(grid1)
    else:
        MMD.extend(list(MMD1))
        AMD.extend(list(AMD1))
        WD.extend(list(WD1))
        SSD.extend(list(SSD1))
        method.extend(list(method1))
        grid.extend(list(grid1))

    return MMD, AMD, WD, SSD, method, grid

def construc_violins(tit,j,k):

    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^2$",j,"./EMULATION/Res_emu_102_clean.txt",[],[],[],[],[],[])
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^3$",j,"./EMULATION/Res_emu_103_clean.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^4$",j,"./EMULATION/Res_emu_104_clean.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^5$",j,"./EMULATION/Res_emu_105_clean.txt",MMD, AMD, WD, SSD, method, grid)

    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^2$",j,"./EMULATION/Res_emu_102_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^3$",j,"./EMULATION/Res_emu_103_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^4$",j,"./EMULATION/Res_emu_104_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^5$",j,"./EMULATION/Res_emu_105_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)

    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^2$",j,"./EMULATION/Res_emu_102_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^3$",j,"./EMULATION/Res_emu_103_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^4$",j,"./EMULATION/Res_emu_104_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)

    df = {"Method": method,
      "Resolution": grid,
      "MMD": MMD,
      "AMD": AMD,
      "SSD": SSD,
      "WD":  WD,
        }

    df = pd.DataFrame(df)

    fsize = 18

    df[df==np.inf]=np.nan
    df = df.dropna()

    plt.subplot(5, 4, 1+k)
    ax = sns.boxplot(x="Resolution", y="MMD", hue="Method",
                    data=df, palette="Set3")
    plt.title(tit, loc='left', fontsize=fsize, fontweight='bold')
    plt.ylim([-12,12])
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\mu_\mathrm{pred}-\mu_\mathrm{sim} \,\, [\sigma_\mathrm{sim}]$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    if k == 0:
        plt.legend(bbox_to_anchor=(0., 1.40), ncol =7,  loc=2, borderaxespad=0.,fontsize=fsize)
    else:
        plt.legend([],[], frameon=False)

    plt.subplot(5, 4, 2+k)
    ax = sns.boxplot(x="Resolution", y="AMD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$|M_\mathrm{pred}-M_\mathrm{sim}|/M_\mathrm{sim} $",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale("log")
    plt.legend([],[], frameon=False)

    plt.subplot(5, 4, 3+k)
    ax = sns.boxplot(x="Resolution", y="SSD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\sigma_\mathrm{pred}/\sigma_{\mathrm{sim}}$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale('log')
    plt.legend([],[], frameon=False)

    plt.subplot(5, 4, 4+k)
    ax = sns.boxplot(x="Resolution", y="WD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\mathrm{Wasserstein \, \, Dist.}$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale('log')
    plt.legend([],[], frameon=False)

def small_plot(tit,j):

    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^2$",j,"./EMULATION/Res_emu_102_clean.txt",[],[],[],[],[],[])
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^3$",j,"./EMULATION/Res_emu_103_clean.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^4$",j,"./EMULATION/Res_emu_104_clean.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("NN",r"$10^5$",j,"./EMULATION/Res_emu_105_clean.txt",MMD, AMD, WD, SSD, method, grid)

    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^2$",j,"./EMULATION/Res_emu_102_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^3$",j,"./EMULATION/Res_emu_103_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^4$",j,"./EMULATION/Res_emu_104_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("MDN",r"$10^5$",j,"./EMULATION/Res_emu_105_clean_MDN.txt",MMD, AMD, WD, SSD, method, grid)

    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^2$",j,"./EMULATION/Res_emu_102_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^3$",j,"./EMULATION/Res_emu_103_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)
    MMD, AMD, WD, SSD, method, grid = appending("GP",r"$10^4$",j,"./EMULATION/Res_emu_104_clean_RBF.txt",MMD, AMD, WD, SSD, method, grid)

    df = {"Method": method,
      "Resolution": grid,
      "MMD": MMD,
      "AMD": AMD,
      "SSD": SSD,
      "WD":  WD,
        }

    df = pd.DataFrame(df)

    fsize = 18

    df[df==np.inf]=np.nan
    df = df.dropna()

    plt.subplot(2, 2, 1)
    ax = sns.boxplot(x="Resolution", y="MMD", hue="Method",
                    data=df, palette="Set3")
    #plt.title(tit, loc='left', fontsize=fsize, fontweight='bold')
    plt.ylim([-12,12])
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\mu_\mathrm{pred}-\mu_\mathrm{sim} \,\, [\sigma_\mathrm{sim}]$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.legend(bbox_to_anchor=(0., 1.15), ncol =7,  loc=2, borderaxespad=0.,fontsize=fsize)
    
    plt.subplot(2, 2, 2)
    ax = sns.boxplot(x="Resolution", y="AMD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$|M_\mathrm{pred}-M_\mathrm{sim}|/M_\mathrm{sim} $",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale("log")
    plt.legend([],[], frameon=False)

    plt.subplot(2, 2, 3)
    ax = sns.boxplot(x="Resolution", y="SSD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\sigma_\mathrm{pred}/\sigma_{\mathrm{sim}}$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale('log')
    plt.legend([],[], frameon=False)

    plt.subplot(2, 2, 4)
    ax = sns.boxplot(x="Resolution", y="WD", hue="Method",
                    data=df, palette="Set3")
    plt.xlabel(r"$\mathrm{Resolution}$",fontsize=fsize)
    plt.ylabel(r"$\mathrm{Wasserstein \, \, Dist.}$",fontsize=fsize)
    plt.xticks(size=fsize)
    plt.yticks(size=fsize)
    plt.yscale('log')
    plt.legend([],[], frameon=False)

Small = True

if not Small:
    plt.figure(2, figsize=(20,20))

    construc_violins("A: GSC",0,0)
    construc_violins("B: GPP",13,4)
    construc_violins("C: GDS",26,8)
    construc_violins("D: Dead",39,12)
    construc_violins("E: Rim",52,16)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.5  )

    cell = "GBM_Emuruns"
    plt.savefig(cell+"_MMD.eps",bbox_inches='tight')

else:
    plt.figure(3, figsize=(15,15))

    small_plot("GSC",0)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.2)

    cell = "GBM_Emuruns"  
    plt.savefig(cell+"_small_MMD.eps",bbox_inches='tight')

#plt.show()


