import numpy as np
cimport numpy as np

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "cpp/py_biperscan.h" namespace "bppc::python":
  cdef cppclass biperscan_minpres_result_t[index_t, grade_t]:
    vector[index_t] col_to_edge;
    vector[index_t] row_to_point;
    vector[grade_t] lens_grades;

    vector[grade_t] minpres_lens_grades;
    vector[grade_t] minpres_distance_grades;
    vector[index_t] minpres_parents;
    vector[index_t] minpres_children;

    double matrix_time;
    double minpres_time;
    

  cdef cppclass biperscan_merge_result_t[index_t, grade_t]:
    vector[index_t] merge_start_columns;
    vector[index_t] merge_end_columns;
    vector[grade_t] merge_lens_grades;
    vector[grade_t] merge_distance_grades;
    vector[index_t] merge_parents;
    vector[index_t] merge_children;
    vector[vector[index_t]] merge_parent_sides;
    vector[vector[index_t]] merge_child_sides;

    double merge_time;


  cdef cppclass biperscan_linkage_result_t[index_t, grade_t]:
    vector[grade_t] linkage_lens_grades;
    vector[grade_t] linkage_distance_grades;
    vector[index_t] linkage_parents;
    vector[index_t] linkage_children;
    vector[index_t] linkage_parent_roots;
    vector[index_t] linkage_child_roots;

    double linkage_time;


  cdef pair[value_t, value_t] minmax_of[value_t](value_t *values_ptr, size_t num_values)


  cdef biperscan_minpres_result_t[index_t, grade_t] biperscan_minpres[index_t, grade_t, dist_t, lens_t](
      dist_t *distances_ptr, size_t num_distances,
      lens_t *point_lens_ptr, size_t num_points
  )


  cdef biperscan_merge_result_t[index_t, grade_t] biperscan_merges[index_t, grade_t](
      grade_t *lens_ptr, grade_t *dist_ptr, index_t *parent_ptr, index_t *child_ptr, 
      size_t num_edges, size_t num_points, size_t min_cluster_size, double limit_fraction
  )

  cdef biperscan_linkage_result_t[index_t, grade_t] biperscan_linkage[index_t, grade_t](
      grade_t *lens_ptr, grade_t *dist_ptr, index_t *parent_ptr, index_t *child_ptr, 
      size_t num_edges, size_t num_points
  )