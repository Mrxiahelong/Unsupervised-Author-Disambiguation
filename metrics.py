import numpy as np
def get_Fscore(labels,y_pred):
    P=labels.reshape((1,-1))
    C=y_pred.reshape((1,-1))
    N=len(C.squeeze())
    p=np.array(list(set(P.squeeze()))).reshape((1,-1))
    c=np.array(list(set(C.squeeze()))).reshape((1,-1))
    P_size=len(p.squeeze())
    try:
        C_size=len(c.squeeze())
    except:
        C_size=1
    Pid=np.double(np.matmul(np.ones((P_size,1)),P)==np.matmul(p.T,np.ones((1,N))))
    Cid=np.double(np.matmul(np.ones((C_size,1)),C)==np.matmul(c.T,np.ones((1,N))))
    CP = np.matmul(Cid,Pid.T)
    Pj=np.sum(CP,axis=0).reshape((1,-1))
    Ci=np.sum(CP,axis=1).reshape((-1,1))
    precision=CP/(np.matmul(Ci,np.ones((1,P_size))))
    recall = CP/( np.matmul(np.ones((C_size,1)),Pj ))
    F=(2* precision*recall)/(precision+recall)
    F[np.isnan(F)]=0
    FMeasure=np.sum(Pj/np.sum(Pj)*np.max(F,axis=0))
    return FMeasure
