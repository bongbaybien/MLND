"""Implement quick sort in Python.
Input a list.
Output a sorted list."""

def quicksort(array):
    if len(array) == 1:
#         print('output array', array, '\n')
        return array
    if len(array) == 2:
        if array[0] >= array[1]:
#             print('output array', [array[1], array[0]], '\n')
            return [array[1], array[0]]
        else:
#             print('output array', array, '\n')
            return array
    else:       
#         print('\n input array', array)
        pivot_index = len(array) - 1
        pivot = array[pivot_index]
#         print('pivot', pivot)
        i = 0
        while i < pivot_index:
#             print('comparison index', i)
            if array[i] >= pivot:
#                 print('pivot_index', pivot_index)
                array[pivot_index] = array[i]
                array[i] = array[pivot_index-1]
                array[pivot_index-1] = pivot
                pivot_index -= 1
#                 print('array', array)
            else:
                i += 1    
        left_array = array[:pivot_index]
        right_array = array[pivot_index:]
        output_array = quicksort(left_array) + quicksort(right_array)
#         print('output array', output_array, '\n')
        return output_array
        

test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
print(quicksort(test))