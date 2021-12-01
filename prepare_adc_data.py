import argparse
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def filter_no_leptons(data):
    is_ele = data[:,1,0] > 23
    is_mu = data[:,5,0] > 23
    is_lep = (is_ele+is_mu) > 0
    data_filtered = data[is_lep]
    return data_filtered

def prepare_data(input_file, events, input_bsm, output_file):
    # read QCD data
    with h5py.File(input_file, 'r') as h5f:
        # remove last feature, which is the type of particle
        data = h5f['Particles'][:,:,:-1]
        np.random.shuffle(data)
        if events==-1: events-=1
        data = data[:events,:,:]
    # remove jets eta >4 or <-4
    data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
    data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
    data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
    data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
    data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
    data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])
    n_before = data.shape[0]
    data = filter_no_leptons(data)
    print('Background before filter',n_before,'after filter',data.shape[0],\
        'cut away',(n_before-data.shape[0])/n_before*100,r'%')
    # fit scaler to the full data
    pt_scaler = StandardScaler()
    data_target = np.copy(data)
    data_target[:,:,0] = pt_scaler.fit_transform(data_target[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))
    # define training, test and validation datasets
    X_train, X_test, Y_train, Y_test = train_test_split(data, data_target, test_size=0.5, shuffle=True)
    del data, data_target

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1)

    # read BSM data
    bsm_data = []

    with h5py.File(input_bsm[0],'r') as h5f_leptoquarks:
        leptoquarks = np.array(h5f_leptoquarks['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        leptoquarks[:,9:19,0] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,0])
        leptoquarks[:,9:19,0] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,0])
        leptoquarks[:,9:19,1] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,1])
        leptoquarks[:,9:19,1] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,1])
        leptoquarks[:,9:19,2] = np.where(leptoquarks[:,9:19,1]>4,0,leptoquarks[:,9:19,2])
        leptoquarks[:,9:19,2] = np.where(leptoquarks[:,9:19,1]<-4,0,leptoquarks[:,9:19,2])
        n_before = leptoquarks.shape[0]
        leptoquarks = filter_no_leptons(leptoquarks)
        print('Leptoquarks before filter',n_before,'after filter',leptoquarks.shape[0],\
            'cut away',(n_before-leptoquarks.shape[0])/n_before*100,r'%')
        leptoquarks = leptoquarks.reshape(leptoquarks.shape[0],leptoquarks.shape[1],leptoquarks.shape[2],1)
        bsm_data.append(leptoquarks)

    with h5py.File(input_bsm[1],'r') as h5f_ato4l:
        ato4l = np.array(h5f_ato4l['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        ato4l[:,9:19,0] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,0])
        ato4l[:,9:19,0] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,0])
        ato4l[:,9:19,1] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,1])
        ato4l[:,9:19,1] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,1])
        ato4l[:,9:19,2] = np.where(ato4l[:,9:19,1]>4,0,ato4l[:,9:19,2])
        ato4l[:,9:19,2] = np.where(ato4l[:,9:19,1]<-4,0,ato4l[:,9:19,2])
        n_before = ato4l.shape[0]
        ato4l = filter_no_leptons(ato4l)
        print('Ato4l before filter',n_before,'after filter',ato4l.shape[0],\
            'cut away',(n_before-ato4l.shape[0])/n_before*100,r'%')
        ato4l = ato4l.reshape(ato4l.shape[0],ato4l.shape[1],ato4l.shape[2],1)
        bsm_data.append(ato4l)

    with h5py.File(input_bsm[2],'r') as h5f_hChToTauNu:
        hChToTauNu = np.array(h5f_hChToTauNu['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        hChToTauNu[:,9:19,0] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,0])
        hChToTauNu[:,9:19,0] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,0])
        hChToTauNu[:,9:19,1] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,1])
        hChToTauNu[:,9:19,1] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,1])
        hChToTauNu[:,9:19,2] = np.where(hChToTauNu[:,9:19,1]>4,0,hChToTauNu[:,9:19,2])
        hChToTauNu[:,9:19,2] = np.where(hChToTauNu[:,9:19,1]<-4,0,hChToTauNu[:,9:19,2])
        n_before = hChToTauNu.shape[0]
        hChToTauNu = filter_no_leptons(hChToTauNu)
        print('hChToTauNu before filter',n_before,'after filter',hChToTauNu.shape[0],\
            'cut away',(n_before-hChToTauNu.shape[0])/n_before*100,r'%')
        hChToTauNu = hChToTauNu.reshape(hChToTauNu.shape[0],hChToTauNu.shape[1],hChToTauNu.shape[2],1)
        bsm_data.append(hChToTauNu)

    with h5py.File(input_bsm[3],'r') as h5f_hToTauTau:
        hToTauTau = np.array(h5f_hToTauTau['Particles'][:,:,:-1])
        # remove jets eta >4 or <-4
        hToTauTau[:,9:19,0] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,0])
        hToTauTau[:,9:19,0] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,0])
        hToTauTau[:,9:19,1] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,1])
        hToTauTau[:,9:19,1] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,1])
        hToTauTau[:,9:19,2] = np.where(hToTauTau[:,9:19,1]>4,0,hToTauTau[:,9:19,2])
        hToTauTau[:,9:19,2] = np.where(hToTauTau[:,9:19,1]<-4,0,hToTauTau[:,9:19,2])
        n_before = hToTauTau.shape[0]
        hToTauTau = filter_no_leptons(hToTauTau)
        print('hToTauTau before filter',n_before,'after filter',hToTauTau.shape[0],\
            'cut away',(n_before-hToTauTau.shape[0])/n_before*100,r'%')
        hToTauTau = hToTauTau.reshape(hToTauTau.shape[0],hToTauTau.shape[1],hToTauTau.shape[2],1)
        bsm_data.append(hToTauTau)

    data = [X_train, Y_train, X_test, Y_test, bsm_data, pt_scaler]

    with open(output_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--input-bsm', type=str, action='append', help='Input file for generated BSM')
    parser.add_argument('--events', type=int, default=-1, help='How many events to proceed')
    parser.add_argument('--output-file', type=str, help='output file', required=True)
    args = parser.parse_args()
    prepare_data(**vars(args))
