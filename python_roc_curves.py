import caffe
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
class PythonROCCurves(caffe.Layer):
    """
    Compute the Accuracy with a Python Layer
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs.")

        self.num_labels = bottom[0].channels
        params = json.loads(self.param_str)
        self.test_iter = params['test_iter']
        
        self.show = 'show' in params and params['show'] or 'no'
        self.savefig = 'savefig' in params and params['savefig'] or ''
        self.figformat = 'figformat' in params and params['figformat'] or 'png'
	self.accumulating = 'accumulating' in params and params['accumulating'] or 'yes'
        
        self.current_iter = 0
        self.savefig_iter = 0
        
        self.n_classes = self.num_labels
        self.y_gt = np.empty((0))
        self.y_score = np.empty((0, self.n_classes))
	self.fig = plt.figure()
        
    def reshape(self, bottom, top):
        # bottom[0] are the net's outputs
        # bottom[1] are the ground truth labels

        # Net outputs and labels must have the same number of elements
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same number of elements.")

        # accuracy output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.current_iter += 1
        # predicted outputs
        pred = np.argmax(bottom[0].data, axis=1)
        accuracy = np.sum(pred == bottom[1].data).astype(np.float32) / bottom[0].num
        top[0].data[...] = accuracy

        self.y_gt = np.append(self.y_gt, bottom[1].data)        
        self.y_score = np.vstack((self.y_score, bottom[0].data))
            
        if self.current_iter == self.test_iter:
            self.current_iter = 0
            
            y_test = np.zeros((self.y_gt.shape[0], self.n_classes))
            for i in range(self.n_classes):
                y_test[np.where(self.y_gt==i),i]=1
                    
            if self.n_classes == 2:
                # Compute ROC curve and ROC area for each class
                fpr, tpr, _ = roc_curve(y_test[:,1], self.y_score[:,1])
                roc_auc = auc(fpr, tpr)
                 
                # Compute micro-average ROC curve and ROC area
                fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), self.y_score.ravel())
                roc_auc_micro = auc(fpr, tpr)
     
                # Plot of a ROC curve for a specific class  
                plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([-0.01, 1.0])
                plt.ylim([0.0, 1.01])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
            else:
                # Compute ROC curve and ROC area for each class
                    
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(self.n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], self.y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                print(self.y_score.shape)
                print(self.y_score.ravel().shape)
                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), self.y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
                
                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(self.n_classes):
                    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
                
                # Finally average it and compute AUC
                mean_tpr /= self.n_classes
                
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                
                # Plot all ROC curves
                plt.plot(fpr["micro"], tpr["micro"],
                         label='micro-average ROC curve (area = {0:0.3f})'
                               ''.format(roc_auc["micro"]),
                         color='deeppink', linestyle=':', linewidth=4)
                
                plt.plot(fpr["macro"], tpr["macro"],
                         label='macro-average ROC curve (area = {0:0.3f})'
                               ''.format(roc_auc["macro"]),
                         color='navy', linestyle=':', linewidth=4)
                
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                for i, color in zip(range(self.n_classes), colors):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label='ROC curve of class {0} (area = {1:0.3f})'
                             ''.format(i, roc_auc[i]))
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([-0.01, 1.0])
                plt.ylim([0.0, 1.01])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curves')
                plt.legend(loc="lower right")
            
            if self.show == 'yes':
                plt.show()
		plt.close(self.fig)
		plt.clf()
                
            if self.savefig != '':
		plt.gcf().savefig(self.savefig+'-'+str(self.savefig_iter)+'.'+self.figformat)
                self.savefig_iter += 1
		plt.clf()
            
	    if self.accumulating != 'yes':
            	self.y_gt = np.empty((0))
            	self.y_score = np.empty((0, self.n_classes))

    def backward(self, top, propagate_down, bottom):
        pass
