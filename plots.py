import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score

class Plots:
    def imgs_dataset(data_dir='./image_dataset', classes=['regular', 'covid'], num_imgs=4, img_width=224, img_height=224):
        item_files = []
        fig, big_axes = plt.subplots(figsize=(15, 12), nrows=3, ncols=1, sharey=True)

        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            all_item_dirs = os.listdir(class_dir)
            item_files = item_files + [os.path.join(class_dir, file) for file in all_item_dirs][:num_imgs]
        
        classes_sp=['saludables', 'con COVID-19']
        for idx, big_ax in enumerate(big_axes, start=0):
            big_ax.set_title("Ultrasonido pulmonar en pacientes %s \n" % classes_sp[idx-1], fontsize=16)
            big_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
            # removes the white frame
            big_ax._frameon = False

        for idx, img_path in enumerate(item_files):
            _ = fig.add_subplot(len(classes), num_imgs, idx + 1)
            img = plt.imread(img_path)
            plt.imshow(img)

        plt.tight_layout()
        plt.savefig("dataset_sample.pdf",bbox_inches='tight')
        plt.show()
        
    def train_loss_acc_results(H, model_dir, fold, epochs):
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, epochs), H.history['loss'], label='train_loss')
        plt.plot(np.arange(0, epochs), H.history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, epochs), H.history['accuracy'], label='train_acc')
        plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='val_acc')
        plt.title('Training Loss and Accuracy on COVID-19 Dataset')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join('{}'.format(model_dir), 'loss_{}e_f{}.png'.format(epochs, fold)))
        plt.show()
        
    def data_to_label(data, text):
        return (np.asarray(["{0:.2f}\n".format(data)+u"\u00B1"+"{0:.2f}".
                            format(text) for data, text in zip(data.flatten(), text.flatten())])).reshape(3,3)
        
    def confusion_matrix_plot(all_cms):
        plt.figure(figsize = (25,6))
        fig = plt.subplot(1,3,1)
        ax = fig.axes
        
        data_abs = np.sum(all_cms, axis=0)
        df_cm = pd.DataFrame(data_abs, index = [i for i in ["COVID-19", "Saludable"]],
                          columns = [i for i in ["COVID-19", "Saludable"]])
        
        sn.set(font_scale=1.5)

        # plt.xticks(np.arange(3)+0.5,("COVID-19", "Pneumonia", "Normal"), rotation=0, fontsize="17", va="center")
        plt.yticks(np.arange(3)+0.5,("COVID-19", "Saludable"), rotation=0, fontsize="17", va="center")
        sn.heatmap(df_cm, annot=True, fmt="g", cmap="YlGnBu")
        ax.xaxis.tick_top()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) 
        plt.xlabel('\nPrediciones', size=25)
        plt.ylabel('Ground truth', size=25)
        plt.title("Valores absolutos\n", size=30,fontweight="bold")
        
        # PRECISION SUBPLOT
        fig = plt.subplot(1,3,2)
        ax = fig.axes
        
        data_prec = all_cms.copy()
        for i in range(5):
            data_prec[i] = data_prec[i]/np.sum(data_prec[i], axis=0)
        prec_stds = np.std(data_prec, axis = 0)
        data_prec = np.mean(data_prec, axis=0)
        labels_prec = Plots.data_to_label(data_prec, prec_stds)
        
        df_cm = pd.DataFrame(data_prec, index = [i for i in ["COVID-19", "Saludable"]],
                  columns = [i for i in ["COVID-19", "Saludable"]])
        sn.set(font_scale=1.5)
        ax.xaxis.tick_top()
        plt.ylabel("Ground truth")
        plt.xlabel("Prediciones")
        plt.title("Precisión")
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) 
        plt.yticks(np.arange(3)+0.5,("COVID-19", "Healthy"), rotation=0, fontsize="17", va="center")
        sn.heatmap(df_cm, annot=labels_prec, fmt='', cmap="YlGnBu")
        plt.xlabel('\nPrediciones', size=25)
        plt.ylabel('Ground truth', size=25)
        plt.title("Precisión\n", size=30,fontweight="bold")
        
        plt.savefig("confusion_matrix_newdata.pdf",bbox_inches='tight') #, bottom=0.2)
        
        #  SENSITIVITY SUBPLOT
        fig = plt.subplot(1,3,3)
        ax = fig.axes
        data_sens = all_cms.copy()
        for i in range(5):
            sums_axis = np.sum(data_sens[i], axis=1)
            data_sens[i] = np.array([data_sens[i,j,:]/sums_axis[j] for j in range(3)])
        sens_stds = np.std(data_sens, axis = 0)
        data_sens = np.mean(data_sens, axis=0)

        labels_sens = Plots.data_to_label(data_sens, sens_stds)
        df_cm = pd.DataFrame(data_sens, index = [i for i in ["COVID-19", "Saludable"]],
                          columns = [i for i in ["COVID-19", "Saludable"]])
        # sn.set(font_scale=1.5)
        
        plt.yticks(np.arange(3)+0.5,("COVID-19", "Saludable"), rotation=0, fontsize="17", va="center")
        #plt.xticks(np.arange(3)+0.5,("COVID-19", "Pneunomia", "Normal"), rotation=0, fontsize="17", va="center")
        ax.xaxis.tick_top()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) 

        sn.heatmap(df_cm, annot=labels_sens, fmt='', cmap="YlGnBu")
        plt.xlabel('\nPrediciones', size=25)
        plt.ylabel('Ground truth', size=25)
        plt.title("Sensibilidad (Recall)\n", size=30,fontweight="bold")

        plt.savefig("confusion_matrix_all.pdf",bbox_inches='tight') #, bottom=0.2)
        
    def roc_auc(saved_logits, saved_gt):
        base_eval_points = np.linspace(0,1,200,endpoint=True)
        data, scores, roc_auc_std = [], [], []
        max_points = []
        for i in range(3):
            out_roc = np.zeros((5, len(base_eval_points)))
            out_prec = np.zeros((5, len(base_eval_points)))

            roc_auc = []
            max_acc = []

            # Iterate over folds
            for k in range(5):
                # get binary predictions for this class
                gt =  (saved_gt[k] == i).astype(int)
                # pred = saved_logits[k][:, i]
                if np.any(saved_logits[k]<0):
                    pred = np.exp(np.array(saved_logits[k]))[:, i]
                else:
                    pred = np.array(saved_logits[k])[:, i]

                roc_auc.append(roc_auc_score(gt, pred))

                precs, recs, fprs, julie_points = [], [], [], []
                for j, thresh in enumerate(np.linspace(0,1.1,100, endpoint=True)):
                    preds_thresholded = (pred>thresh).astype(int)
                    tp = np.sum(preds_thresholded[gt==1])
                    p = np.sum(gt)
                    n = len(gt)-p
                    fp = np.sum(preds_thresholded[gt==0])
                    inverted = np.absolute(preds_thresholded - 1)
                    tn = np.sum(inverted[gt==0])
                    fn = np.sum(inverted[gt==1])
                    fpr = fp/float(n)
                    tpr = tp/float(p)

                    if tp+fp ==0:
                        precs.append(1)
                    else:
                        precs.append(tp/(tp+fp))
                    recs.append(tpr)
                    fprs.append(fpr)
                    julie_points.append((tp+tn)/(tp+tn+fp+fn))

                # clean
                recs = np.asarray(recs)
                precs = np.asarray(precs)
                fprs = np.asarray(fprs)
                sorted_inds = np.argsort(recs)
                # prepare for precision-recall curve
                precs_sorted = precs[sorted_inds]
                recs_sorted = recs[sorted_inds]
                precs_cleaned = precs_sorted[recs_sorted>0]
                recs_cleaned = recs_sorted[recs_sorted>0]
                precs_inter = np.interp(base_eval_points, recs_cleaned, precs_cleaned)
                # prepare for roc-auc curve
                sorted_inds = np.argsort(fprs)
                recs_fpr_sorted = recs[sorted_inds]
                fprs_sorted = fprs[sorted_inds]
                roc_inter = np.interp(base_eval_points, fprs_sorted, recs_fpr_sorted)
                # append current fold
                out_prec[k] = precs_inter
                out_roc[k] = roc_inter

                # compute recall of max acc:
                max_acc.append(recs[np.argmax(julie_points)])

            # out_curve = np.mean(np.asarray(out_curve), axis=0)

            prec_mean = np.mean(out_prec, axis=0)
            prec_std = np.std(out_prec, axis=0)
            roc_mean = np.mean(out_roc, axis=0)
            roc_std = np.std(out_roc, axis=0)

            # append scores
            scores.append(round(np.mean(roc_auc),2))
            roc_auc_std.append(round(np.std(roc_auc),2))

            # point of maximum accuracy
            max_points.append(np.mean(max_acc))

            data.append((roc_mean, roc_std, prec_mean, prec_std))
        return data, max_points, scores, roc_auc_std

    def closest(in_list, point):
        return np.argmin(np.absolute(np.asarray(in_list)-point))
    
    def roc_sensibility_comparison(cols, classes, data, max_points, scores, roc_auc_std):
        base_eval_points = np.linspace(0,1,200,endpoint=True)
        plt.figure(figsize=(6,5))
        plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
        for i in range(3):
            roc_mean, roc_std, _, _ = data[i]
            lab = classes[i]+" (%.2f"%scores[i]+"$\pm$"+str(roc_auc_std[i])+")"
            plt.plot(base_eval_points, roc_mean, 'k-', c=cols[i], label=lab, lw=3)
            # print(len(r), max_points[i])
            # print(base_eval_points[Plots.closest(roc_mean, max_points[i])], max_points[i])
            plt.scatter(base_eval_points[Plots.closest(roc_mean, max_points[i])], max_points[i], s=150, marker="o", c=cols[i])
            plt.fill_between(base_eval_points, roc_mean-roc_std, roc_mean+roc_std, alpha=0.1, facecolor=cols[i])
            plt.ylim(0,1.03)
        plt.xlim(-0.02,1)
        plt.ylabel("$\\bf{Sensibilidad}$", fontsize=20)
        plt.xlabel("$\\bf{Radio \ de\ falsos\ positivos}$", fontsize=20)
        plt.legend(fontsize=18, title="    $\\bf{Clase}\ \\bf(ROC-AUC)}$") # "\n  $\\bf{(o:\ maximal\ accuracy)}$")
        # plt.title("$\\bf{ROC\ curves}$", fontsize=15)
        plt.savefig("./roc_curves_cam.pdf", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        
    def roc_presicion_comparison(cols, classes, data, max_points, scores, roc_auc_std):
        base_eval_points = np.linspace(0,1,200,endpoint=True)
        plt.figure(figsize=(6,5))
        plt.plot([1, 0], [0, 1], color='grey', lw=1.5, linestyle='--')
        for i in range(3):
            _, _, prec_mean, prec_std = data[i]
            # prec_cleaned = prec[rec>0]
            # rec_cleaned = rec[rec>0]
            # s2_cleaned = s2[rec>0]
            lab = classes[i] # +" (%.2f"%scores[i]+"$\pm$"+str(roc_auc_std[i])+")"
            plt.plot(base_eval_points, prec_mean, 'k-', c=cols[i], label=lab, lw=3)
            plt.fill_between(base_eval_points, prec_mean-prec_std, prec_mean+prec_std, alpha=0.1, facecolor=cols[i])
        plt.ylim(0,1.03)
        plt.xlim(-0.02,1)
        plt.ylabel("$\\bf{Precisión}$", fontsize=20)
        plt.xlabel("$\\bf{Sensibilidad (recall)}$", fontsize=20)
        plt.legend(fontsize=18, title="    $\\bf{Clase}$") # "\n  $\\bf{(o:\ maximal\ accuracy)}$")
        # plt.title("$\\bf{ROC\ curves}$", fontsize=15)
        plt.savefig("./prec_rec_curves_cam.pdf", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
     