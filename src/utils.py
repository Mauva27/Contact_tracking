import numpy as np
import os
import sys
import pickle

def find_nan(data):
    nan_li = []
    for i in range(len(data)):
        if np.isnan(data[i]).any():
            nan_li.append(i)
    nan_li = np.array(nan_li)
    new_data = np.delete(np.array(data),nan_li,0)
    return new_data,nan_li

def create_folder(directory):
    if not os.path.exists(directory + 'data'):
        os.mkdir(directory + 'data')
        print ("data directory has been created")

def saving_pickle(directory,filename,data):
    create_folder(directory)
    pickle.dump(data,open(directory + 'data/{}.p'.format(filename), 'wb'))

def saving_xyz(directory,filename,data,type,which):
    create_folder(directory)
    # ipdb.set_trace()
    f = open(directory + 'data/' + '{}.xyz'.format(filename), 'w')
    f.close()

    if type == 'xyz':
        for t in data:
            with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                file.write('{}\nProperties=species:S:1:pos:R:3:size:C:1 Time={}\n'.format(data[t][which].shape[0],t))
            output = np.zeros((data[t][which].shape[0],4))
            output[:,0] = data[t][which][:,0]
            output[:,1] = data[t][which][:,1]
            output[:,2] = data[t][which][:,2]
            output[:,3] = data[t][which][:,3]

            for row in output:
                re = row.reshape(1,4)
                with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                    np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f' for i in range(output.shape[1] - 1)] ))

def saving_binary_xyz(directory,filename,data1,data2,type,which1,which2):
    create_folder(directory)
    # ipdb.set_trace()
    f = open(directory + 'data/' + '{}.xyz'.format(filename), 'w')
    f.close()

    assert len(data1) == len(data2), ''

    if type == 'xyz':
        for t in data1:
            with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
                file.write('{}\nProperties=species:S:2:pos:R:3:size:C:1 Time={}\n'.format(data1[t][which1].shape[0]+data2[t][which2].shape[0],t))
            output = np.zeros((data1[t][which1].shape[0],4))
            output[:,0] = data1[t][which1][:,0]
            output[:,1] = data1[t][which1][:,1]
            output[:,2] = data1[t][which1][:,2]
            output[:,3] = data1[t][which1][:,3]

            for row in output:
                re = row.reshape(1,4)
                with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                    np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f' for i in range(output.shape[1] - 1)] ))

            output_ = np.zeros((data2[t][which2].shape[0],4))
            output_[:,0] = data2[t][which2][:,0]
            output_[:,1] = data2[t][which2][:,1]
            output_[:,2] = data2[t][which2][:,2]
            output_[:,3] = data2[t][which2][:,3]

            for row in output_:
                re = row.reshape(1,4)
                with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
                    np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['B\t%1.5f'] + ['%1.5f' for i in range(output_.shape[1] - 1)] ))

    # elif type == 'array':
    #     for f in data:
    #         with open(directory + 'data/' + '{}.xyz'.format(filename), 'a') as file:
    #             file.write('{}\nProperties=species:S:1:pos:R:3:size:C:1 Time=0\n'.format(data.shape[0]))
    #         output = np.zeros((data.shape[0],4))
    #         output[:,0] = data[:,0]
    #         output[:,1] = data[:,1]
    #         output[:,2] = data[:,2]
    #         output[:,3] = data[:,3]
    #
    #         for row in output:
    #             re = row.reshape(1,4)
    #             with open(directory + 'data/' +'{}.xyz'.format(filename), 'a') as this_row:
    #                 np.savetxt(this_row, re, delimiter = '\t', fmt = ' '.join(['A\t%1.5f'] + ['%1.5f' for i in range(output.shape[1] - 1)] ))
