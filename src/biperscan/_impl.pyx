# distutils: language = c++
import numpy  as np
cimport numpy as np

cimport cython
from cython.view cimport array as cvarray
from ._impl cimport (
  minmax_of as cpp_minmax_of, 
  biperscan_minpres as cpp_biperscan_minpres,
  biperscan_merges as cpp_biperscan_merges,
  biperscan_linkage as cpp_biperscan_linkage
)

ctypedef fused value_t:
  np.float32_t
  np.float64_t


cdef class UnsignedWrapper:
  """Wraps a vector of np.uint32_t values for numpy to use."""
  cdef vector[np.uint32_t] vec
  cdef Py_ssize_t shape[1]
  cdef Py_ssize_t strides[1]
  

  def __getbuffer__(self, Py_buffer *buffer, int flags):
    if buffer == NULL:
      raise ValueError("NULL view in __getbuffer__")

    buffer.buf = &(self.vec[0])
    buffer.obj = <object>self
    buffer.len = self.vec.size() * sizeof(self.vec[0])
    buffer.itemsize = sizeof(self.vec[0])
    buffer.readonly = 0
    buffer.ndim = 1
    buffer.format = 'I' # np.uint32_t
    buffer.shape = self.shape
    buffer.strides = self.strides
    buffer.internal = NULL
    buffer.suboffsets = NULL

  cdef as_array(self, vector[np.uint32_t] &data):
    self.vec.swap(data)
    self.shape[0] = self.vec.size()
    self.strides[0] = sizeof(np.uint32_t)  
    arr = np.asarray(self, copy=False)
    assert arr.shape[0] == self.vec.size()
    return arr
  

cdef class FloatWrapper:
  """Wraps a vector of np.float32_t values for numpy to use."""
  cdef vector[np.float32_t] vec
  cdef Py_ssize_t shape[1]
  cdef Py_ssize_t strides[1] 

  def __getbuffer__(self, Py_buffer *buffer, int flags):
    if buffer == NULL:
      raise ValueError("NULL view in __getbuffer__")

    buffer.buf = &(self.vec[0])
    buffer.obj = <object>self
    buffer.len = self.vec.size() * sizeof(self.vec[0])
    buffer.itemsize = sizeof(self.vec[0])
    buffer.readonly = 0
    buffer.ndim = 1
    buffer.format = 'f' # np.float32_t
    buffer.shape = self.shape
    buffer.strides = self.strides
    buffer.internal = NULL
    buffer.suboffsets = NULL

  cdef as_array(self, vector[np.float32_t] &data):
    self.vec.swap(data)
    self.shape[0] = self.vec.size()
    self.strides[0] = sizeof(np.float32_t)
    arr = np.asarray(self, copy=False)
    assert arr.shape[0] == self.vec.size()
    return arr
  

@cython.boundscheck(False)
@cython.wraparound(False)
def minmax_of(value_t[::1] values not None):
  """
  Computes an array's minimum and maximum value.

  Parameters
  ----------
  values : list, numpy array, cython array
    The list of values to compute the minimum and maximum values for.


  Returns
  -------
  A tuple with (min, max)
  """
  cdef size_t num_values = len(values)
  cdef pair[value_t, value_t] extrema = cpp_minmax_of(<value_t *>&values[0], num_values)
  return (extrema.first, extrema.second)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_minimal_presentation(np.ndarray[np.float32_t, ndim=1] distances not None, 
                                 np.ndarray[np.float32_t, ndim=1] point_lens not None): 
  """
  Constructs a minimal presentation from the given distances and point lens.

  Parameters
  ----------
  distances : list, numpy array, cython array
    A condensed distance matrix.
  
  point_lens : list, numpy array, cython array
    A point-lens value for each data point.

  Returns
  -------
    1. An array mapping columns to distance indices.
    2. An array mapping rows to data point indices.
    3. An array with the data points lens grades.
    4. A dictionary with arrays describing the minimal presentation's edges.
  """
  # Call the function
  cdef size_t num_points = len(point_lens)
  cdef size_t num_edges = len(distances)
  cdef biperscan_minpres_result_t[np.uint32_t, np.uint32_t] res = cpp_biperscan_minpres[np.uint32_t, np.uint32_t, np.float32_t, np.float32_t](
    <np.float32_t *>&distances[0], num_edges, <np.float32_t *>&point_lens[0], num_points
  )
  
  # Decode arrays
  cdef np.ndarray[np.uint32_t, ndim=1] col_to_edge = UnsignedWrapper().as_array(res.col_to_edge)
  cdef np.ndarray[np.uint32_t, ndim=1] row_to_point = UnsignedWrapper().as_array(res.row_to_point)

  # Decode minimal presentation
  cdef np.ndarray[np.uint32_t, ndim=1] lens_grades = UnsignedWrapper().as_array(res.lens_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] minpres_lens_grade = UnsignedWrapper().as_array(res.minpres_lens_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] minpres_distance_grade = UnsignedWrapper().as_array(res.minpres_distance_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] minpres_parent = UnsignedWrapper().as_array(res.minpres_parents)
  cdef np.ndarray[np.uint32_t, ndim=1] minpres_child = UnsignedWrapper().as_array(res.minpres_children)
  minpres = dict(
    lens_grade=minpres_lens_grade, 
    distance_grade=minpres_distance_grade, 
    parent=minpres_parent, 
    child=minpres_child,
  )

  # Decode times
  cdef double matrix_time = res.matrix_time
  cdef double minpres_time = res.minpres_time

  # Return result
  return (col_to_edge, row_to_point, lens_grades, minpres, matrix_time, minpres_time)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_minpres_merges(dict minpres not None, 
                           num_points not None, 
                           min_cluster_size not None, 
                           limit_fraction not None):
  """
  Extracts merges from the given minimal presentation.

  Parameters
  ----------
  minpres : dict
    The minimal presentation as returned by `compute_minimal_presentation`.
  
  num_points : int
    The number of data points.

  min_cluster_size : int
    The minimum number of points in a cluster.

  limit_fraction : float
    The maximum distance grade's fraction to use a upper limit.

  Returns
  -------
  merges : dict
    The detected merges as a dictionary of arrays.
  """
  cdef size_t num_edges = len(minpres['parent'])
  cdef np.uint32_t[::1] minpres_lens_grades = minpres['lens_grade']
  cdef np.uint32_t[::1] minpres_distance_grades = minpres['distance_grade']
  cdef np.uint32_t[::1] minpres_parents = minpres['parent']
  cdef np.uint32_t[::1] minpres_children = minpres['child']
  cdef biperscan_merge_result_t[np.uint32_t, np.uint32_t] res = cpp_biperscan_merges[np.uint32_t, np.uint32_t](
    <np.uint32_t *>&minpres_lens_grades[0], <np.uint32_t *>&minpres_distance_grades[0], 
    <np.uint32_t *>&minpres_parents[0], <np.uint32_t *>&minpres_children[0], 
    num_edges, num_points, min_cluster_size, limit_fraction
  )
  
  # Decode hierarchy
  cdef np.ndarray[np.uint32_t, ndim=1] merge_start_column = UnsignedWrapper().as_array(res.merge_start_columns)
  cdef np.ndarray[np.uint32_t, ndim=1] merge_end_column = UnsignedWrapper().as_array(res.merge_end_columns)
  cdef np.ndarray[np.uint32_t, ndim=1] merge_lens_grade = UnsignedWrapper().as_array(res.merge_lens_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] merge_distance_grade = UnsignedWrapper().as_array(res.merge_distance_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] merge_parent = UnsignedWrapper().as_array(res.merge_parents)
  cdef np.ndarray[np.uint32_t, ndim=1] merge_child = UnsignedWrapper().as_array(res.merge_children)
  cdef list merge_parent_side = [
    UnsignedWrapper().as_array(res.merge_parent_sides[i]) 
    for i in range(len(res.merge_parent_sides))
  ]
  cdef list merge_child_side = [
    UnsignedWrapper().as_array(res.merge_child_sides[i]) 
    for i in range(len(res.merge_child_sides))
  ]
  merges = dict(
    start_column=merge_start_column,
    end_column=merge_end_column,
    lens_grade=merge_lens_grade,
    distance_grade=merge_distance_grade,
    parent=merge_parent,
    child=merge_child,
    parent_side=merge_parent_side,
    child_side=merge_child_side,
  )

  # Decode times
  cdef double merge_time = res.merge_time

  # Return result
  return merges, merge_time


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_linkage_hierarchy(dict minpres not None, 
                              np.ndarray[np.uint32_t, ndim=1] row_to_point not None):
  """
  Constructs a linkage hierarchy from the given minimal presentation.

  Parameters
  ----------
  minpres : dict
    The minimal presentation as returned by `compute_minimal_presentation`.

  row_to_point : list, numpy array, cython array
    A mapping from rows to data points.

  Returns
  -------
  hierarchy : dict
    The linkage hierarchy as a dictionary of arrays.
  """
  cdef size_t num_edges = len(minpres['parent'])
  cdef size_t num_points = len(row_to_point)
  cdef np.uint32_t[::1] minpres_lens_grades = minpres['lens_grade']
  cdef np.uint32_t[::1] minpres_distance_grades = minpres['distance_grade']
  cdef np.uint32_t[::1] minpres_parents = minpres['parent']
  cdef np.uint32_t[::1] minpres_children = minpres['child']
  cdef biperscan_linkage_result_t[np.uint32_t, np.uint32_t] res = cpp_biperscan_linkage[np.uint32_t, np.uint32_t](
    <np.uint32_t *>&minpres_lens_grades[0], <np.uint32_t *>&minpres_distance_grades[0], 
    <np.uint32_t *>&minpres_parents[0], <np.uint32_t *>&minpres_children[0], 
    num_edges, num_points
  )
  
  # Decode hierarchy
  cdef np.uint32_t value, idx;
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_lens_grade = UnsignedWrapper().as_array(res.linkage_lens_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_distance_grade = UnsignedWrapper().as_array(res.linkage_distance_grades)
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_parent = UnsignedWrapper().as_array(res.linkage_parents)
  for idx, value in enumerate(linkage_parent):
    if value < num_points:
      linkage_parent[idx] = row_to_point[value]
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_child = UnsignedWrapper().as_array(res.linkage_children)
  for idx, value in enumerate(linkage_child):
    if value < num_points:
      linkage_child[idx] = row_to_point[value]
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_parent_root = UnsignedWrapper().as_array(res.linkage_parent_roots)
  cdef np.ndarray[np.uint32_t, ndim=1] linkage_child_root = UnsignedWrapper().as_array(res.linkage_child_roots)
  hierarchy = dict(
    lens_grade=linkage_lens_grade,
    distance_grade=linkage_distance_grade,
    parent=linkage_parent,
    child=linkage_child,
    parent_root=linkage_parent_root,
    child_root=linkage_child_root,
  )

  # Decode times
  cdef double linkage_time = res.linkage_time

  # Return result
  return hierarchy, linkage_time
  
  
@cython.boundscheck(False)
@cython.wraparound(False)
def mutual_reachability_from_pdist(
  np.ndarray[np.double_t, ndim=1] core_distances,
  np.ndarray[np.double_t, ndim=1] dists, 
  np.intp_t dim
):
  """https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/_hdbscan_reachability.pyx"""
  cdef np.intp_t i
  cdef np.intp_t j
  cdef np.intp_t result_pos

  result_pos = 0
  for i in range(dim):
    for j in range(i + 1, dim):
      if core_distances[i] > core_distances[j]:
        if core_distances[i] > dists[result_pos]:
          dists[result_pos] = core_distances[i]

      else:
        if core_distances[j] > dists[result_pos]:
          dists[result_pos] = core_distances[j]

      result_pos += 1

  return dists