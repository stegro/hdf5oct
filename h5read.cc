/*
 *
 *    Copyright (C) 2012 Tom Mullins
 *    Copyright (C) 2015 Tom Mullins, Thorsten Liebig, Anton Starikov, Stefan Großhauser
 *    Copyright (C) 2008-2013 Andrew Collette
 *
 *    This file is part of hdf5oct.
 *
 *    hdf5oct is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    hdf5oct is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with hdf5oct.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
// integrated into the GNU Octave build
#include "oct.h"
#include "lo-ieee.h"
#else
// as a package
#include <octave/oct.h>
#include <octave/lo-ieee.h>
#endif

#include <cstdlib>
#include <cerrno>
#include <iostream>
#include <algorithm>
#include <string>
#include "gripes.h"
#include "file-stat.h"

using namespace std;

#if ((H5_VERS_MAJOR > 1) || (H5_VERS_MINOR >= 8))
// define this in case there is no configure script at work. This
// should not be necessary any more when integrated into core.
#define HAVE_HDF5_18 1
#endif

#if defined (HAVE_HDF5) && defined (HAVE_HDF5_18)
#include <hdf5.h>
#include "h5read.h"

#include "ls-hdf5.h"

bool
any_int_leq_zero (const Matrix& mat)
{
  for (int i = 0; i < mat.numel (); i++)
    {
      if (mat(i) < 0.5)
        return true;
    }
  return false;
}

// if ALLOW_ZEROS, then Infs will be converted to 0s (used for COUNT)
// else, 0s will produce an error message (used for others)
int
check_vec (const octave_value& val, Matrix& mat/*out*/,
           const char *name, bool allow_zeros)
{
  mat = val.matrix_value ();
  if (error_state)
    return 0;

  if (! mat.is_vector ())
    {
      error ("%s must be a vector", name);
      return 0;
    }

  double mind, maxd;
  if (allow_zeros)
    {
      for (int i = 0; i < mat.numel (); i++)
        {
          if (mat(i) == octave_Inf)
            mat(i) = 0;
        }
      if (! mat.all_integers (mind, maxd) || mat.any_element_is_negative ())
        {
          error ("%s can only contain non-negative integers", name);
          return 0;
        }
    }
  else if (! mat.all_integers (mind, maxd) || any_int_leq_zero (mat))
    {
      error ("%s can only contain positive integers", name);
      return 0;
    } 

  return 1;
}

#endif

DEFUN_DLD (h5read, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{data} =} h5read (@var{filename}, @var{dsetname})\n\
@deftypefnx {Loadable Function} {@var{data} =} h5read (@var{filename}, @var{dsetname}, @var{start}, @var{count})\n\
@deftypefnx {Loadable Function} {@var{data} =} h5read (@var{filename}, @var{dsetname}, @var{start}, @var{count}, @var{stride})\n\
@deftypefnx {Loadable Function} {@var{data} =} h5read (@var{filename}, @var{dsetname}, @var{start}, @var{count}, @var{stride}, @var{block})\n\
Read a hyperslab of data from an HDF5 file specified by its @var{filename}. \n\
The datatype will be coerced to double.\n\
For example:\n\
\n\
@example\n\
@group\n\
data = h5read (\"mydata.h5\", \"/grid/time\");\n\
@end group\n\
@end example\n\
\n\
The variable @var{dsetname} is the name of the dataset in the HDF5\n\
file to read. It has to be specified by its absolute path (or relative\n\
to the root group / ).\n\
\n\
The other four arguments are 1xn or nx1 matrices, where n is the number of dimensions\n\
in the dataset. If all are omitted, the entire dataset will be read.\n\
\n\
@var{start} is a 1-based starting offset\n\
@var{count} is the number of blocks to read. If 0 or Inf\n\
is specified in any dimension, as many blocks as possible \n\
are read in that dimension (this is not valid for @code{h5write}).\n\
@var{stride} is the offset between the start of each block. \n\
Defaults to a vector of ones.\n\
@var{block} is the size of each block to read. Defaults to a vector of ones.\n\
\n\
Datasets having a compound type consisting of two double values will\n\
be interpreted as complex valued.\n\
\n\
Generally this function tries to use the Octave datatype of\n\
the appropriate size for the given HDF5 type.\n\
\n\
@seealso{h5write}\n\
@end deftypefn")
{
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature ("h5read", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();

  if (nargin < 2 || nargin == 3 || nargin > 6 || nargout > 1)
    {
      print_usage ();
      return octave_value ();
    }
  if (! (args(0).is_string () && args (1).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }

  string filename = args(0).string_value ();
  string dsetname = args(1).string_value ();
  if (error_state)
    return octave_value ();

  //open the hdf5 file
  H5File file (filename.c_str (), false, false);
  if (error_state)
    return octave_value ();

  if (nargin < 4)
    {
      octave_value retval = file.read_dset_complete (dsetname.c_str ());
      return retval;
    }
  else
    {
      Matrix start, count, stride, block;
      int err = 0;
    
      err = err || ! check_vec (args(2), start, "START", false);
      start -= 1;

      err = err || ! check_vec (args(3), count, "COUNT", true);

      if (nargin < 5)
        stride = Matrix ();
      else
        err = err || ! check_vec (args(4), stride, "STRIDE", false);

      if (nargin < 6)
        block = Matrix ();
      else
        err = err || ! check_vec (args(5), block, "BLOCK", false);

      if (err)
        return octave_value ();

      return file.read_dset_hyperslab (dsetname.c_str (),
                                       start, count, stride, block, nargin-2);
    }
#endif
}

DEFUN_DLD (h5readatt, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{data} =} h5readatt (@var{filename}, @var{objectname}, @var{attname})\n\
\n\
Reads one attribute of an object from an HDF5 file, specified by the\n\
@var{filename} and the @var{objectname}.\n\
The third argument @var{attname} is the name of the attribute which \n\
is to read.\n\
\n\
@seealso{h5writeatt}\n\
@end deftypefn")
{
  octave_value retval;
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature ("h5readatt", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();
  if (nargin != 3)
    {
      print_usage ();
      return retval;
    }
  if (! (args(0).is_string () && args(1).is_string () && args(2).is_string ()))
    {
      print_usage ();
      return retval;
    }

  string filename = args(0).string_value ();
  string objname = args(1).string_value ();
  string attname = args(2).string_value ();
  if (error_state)
    return octave_value ();
  
  //open the hdf5 file
  H5File file (filename.c_str (), false, false);
  if (error_state)
    return octave_value ();
        
  retval = file.read_att (objname.c_str (), attname.c_str ());
  return retval;

#endif
}


DEFUN_DLD (h5write, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} h5write (@var{filename}, @var{dsetname}, @var{data})\n\
@deftypefnx {Loadable Function} h5write (@var{filename}, @var{dsetname}, @var{data}, @var{start}, @var{count})\n\
@deftypefnx {Loadable Function} h5write (@var{filename}, @var{dsetname}, @var{data}, @var{start}, @var{count}, @var{stride})\n\
@deftypefnx {Loadable Function} h5write (@var{filename}, @var{dsetname}, @var{data}, @var{start}, @var{count}, @var{stride}, @var{block})\n\
\n\
Write a matrix @var{data} to the specified location @var{dsetname} in \n\
a HDF5 file specified by @var{filename}.\n\
\n\
In the  simplest form of this function, if the file @var{filename} does\n                                       \
not exist, it will be created and if the dataset @var{dsetname} already\n\
exists, it will be overwritten.\n\
\n\
If more than three arguments are specified, an error will be raised if\n\
the file @var{filename} or the dataset @var{dsetname} do not exist\n\
(use @code{h5create} in order to create an file or an empty dataset).\n \
If the dataset exists, the data will be written into the specified \n\
hyperslab. This way it is possible to append data to existing datasets,\n\
provided their maximum size is sufficiently large.\n\
\n\
@var{start} is a 1-based starting offset\n\
@var{count} is the number of blocks to read. In view of @code{h5read}, note that 0 or Inf\n\
are not valid elements here.\n\
@var{stride} is the offset between the start of each block. \n\
Defaults to a vector of ones.\n\
@var{block} is the size of each block to read. Defaults to a vector of ones.\n\
\n\
Complex valued data will lead to datasets having a compound type consisting\n\
of two double values.\n\
\n\
Generally this function tries to use the HDF5 datatype of\n\
the appropriate size for the given Octave type.\n\
\n\
@seealso{h5read}\n\
@end deftypefn")
{
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature ("h5write", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();

  if (! (nargin == 3 || nargin == 5 || nargin  == 6 || nargin == 7) || nargout != 0)
    {
      print_usage ();
      return octave_value ();
    }
  if (! (args(0).is_string () && args(1).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }
  string filename = args(0).string_value ();
  string location = args(1).string_value ();
  
  if (error_state)
    return octave_value ();
    

  if (nargin == 3)
    {
      //open the hdf5 file, create it if it does not exist
      H5File file (filename.c_str (), true, true);
      if (error_state)
        return octave_value ();
      file.write_dset (location.c_str (),
                       args(2));
    }
  else  
    {
      //open the hdf5 file, complain if it does not exist
      H5File file (filename.c_str (), false, true);
      if (error_state)
        return octave_value ();

      Matrix start, count, stride, block;
      int err = 0;
      err = err || ! check_vec (args(3), start, "START", false);
      start -= 1;

      // A count value 0 is not allowed when writing data.
      err = err || ! check_vec (args(4), count, "COUNT", false);

      if (nargin <= 5)
        stride = Matrix ();
      else
        err = err || ! check_vec (args(5), stride, "STRIDE", false);

      if (nargin <= 6)
        block = Matrix ();
      else
        err = err || ! check_vec (args(6), block, "BLOCK", false);

      if (err)
        return octave_value ();

      file.write_dset_hyperslab (location.c_str (),
                                 args(2),
                                 start, count, stride, block, nargin-3);
    }

  return octave_value ();
#endif
}


DEFUN_DLD (h5writeatt, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} h5writeatt (@var{filename}, @var{objectname}, @var{attname}, @var{attvalue})\n\
\n\
Write an attribute with name @var{attname} and value @var{attvalue} to\n\
the object named @var{objectname} in the HDF5 file specified by @var{filename}.\n\
\n\
@seealso{h5readatt}\n\
@end deftypefn")
{
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature ("h5writeatt", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();

  if (nargin != 4 || nargout != 0)
    {
      print_usage ();
      return octave_value ();
    }
  if (! (args(0).is_string () && args(1).is_string () && args(2).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }

  string filename = args(0).string_value ();
  string location = args(1).string_value ();
  string attname = args(2).string_value ();
  
  if (error_state)
    return octave_value ();
    
  //open the hdf5 file
  H5File file (filename.c_str (), false, true);
  if (error_state)
    return octave_value ();

  file.write_att (location.c_str (), attname.c_str (),
                  args(3));

  return octave_value ();
#endif
}


DEFUN_DLD (h5create, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} h5create (@var{filename}, @var{dsetname}, @var{size}, @var{key}, @var{val},...)\n\
\n\
Create a dataset with name @var{dsetname} and size @var{size} \n\
in the HDF5 file specified by @var{filename}.  Intermediate groups\n\
are created as necessary.\n\
\n\
The vector @var{size} may contain one or several Inf (or \n\
equivalently: zero) values.\n\
This will lead to unlimited maximum extent of the dataset in the\n\
respective dimensions and 0 initial extent.\n\
Note that any dataset with at least one unlimited dimension must be chunked and\n\
it is generally recommended for large datasets.\n\
\n\
The list of @var{key}, @var{val} arguments allows to specify\n\
certain properties of the dataset. Allowed settings are:\n\
\n\
@table @asis\n\
@item @option{Datatype}\n\
one of the strings @samp{double} @samp{single} @samp{uint64} @samp{uint32} @samp{uint16} @samp{uint8} @samp{int64} @samp{int32} @samp{int16} @samp{int8} \n\
\n\
@item @option{ChunkSize}\n\
The value may be either a vector specifying the chunk size,\n\
or an empty vector [], which means no chunking (this is the default),\n\
or the string @samp{auto} which makes the library choose automatically \n\
an appropriate chunk size, as best as it can. Note that the @samp{auto}\n\
setting is not @sc{matlab} compatible.\n\
@end table\n\
\n\
@seealso{h5write}\n\
@end deftypefn")
{
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature("h5create", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();

  if (! (nargin ==  3 || nargin == 5 || nargin == 7) || nargout != 0)
    {
      print_usage ();
      return octave_value ();
    }
  if (! (args(0).is_string () && args(1).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }
  if ((nargin == 5 && ! args(3).is_string ())
      || (nargin == 7 && ! args(5).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }
  string filename = args(0).string_value ();
  string location = args(1).string_value ();
  if (error_state)
    return octave_value ();
  
  Matrix size;
  if (! check_vec (args(2), size, "SIZE", true))
    return octave_value ();


  // loop over the key-value pairs and see what is given
  string datatype = "double";
  Matrix chunksize;
  for (int i = 3; i+1 < nargin; i+=2)
    {
      if (args(i).string_value () == "Datatype")
        {
          datatype = args(i+1).string_value ();
          if (error_state)
            {
              error ("Datatype argument must be a string");
              return octave_value ();
            }
        }
      else if (args(i).string_value () == "ChunkSize")
        {
	  if (args(i+1).is_string ())
	    {
	      if(args(i+1).string_value () != "auto")
		{
		  error ("ChunkSize argument must be either a vector, or the string 'auto'.");
		  return octave_value ();
		}
	      chunksize = args(2).matrix_value ();
	      chunksize(0) = 0;

	    }
          else if (! check_vec (args(i+1), chunksize, "ChunkSize", false))
            return octave_value ();
        }
      else
        {
          error ("unknown parameter name %s", args(i).string_value ().c_str ());
          return octave_value ();
        }
    }

  
  
  //open the hdf5 file
  H5File file (filename.c_str (), true, true);
  if (error_state)
    return octave_value ();
  file.create_dset (location.c_str (), size, datatype.c_str (), chunksize);
  
  return octave_value ();
#endif
}


DEFUN_DLD (h5delete, args, nargout,
           "-*- texinfo -*-\n\
@deftypefn {Loadable Function} h5delete (@var{filename}, @var{objname})\n\
@deftypefnx {Loadable Function} h5delete (@var{filename}, @var{objname}, @var{attname})\n\
\n\
In the first form, delete a dataset or group with name @var{objname}\n\
in the HDF5 file specified by @var{filename}. Note that hdf5 is like a\n\
filesystem: the library does not free the used space when a dataset is\n\
deleted. One can use the tool h5repack afterwards to actually reduce the \n\
filesize.\n                                                             \
\n\
In the second form, delete an attribute with name @var{attname} associated\n\
to a dataset or group with name @var{objname}\n\
in the HDF5 file specified by @var{filename}.\n\
\n\
Note that this function is not @sc{matlab} compliant.\n\
\n\
@seealso{h5create}\n\
@end deftypefn")
{
#if ! (defined (HAVE_HDF5) && defined (HAVE_HDF5_18))
  err_disabled_feature("h5delete", "HDF5 IO");
  return octave_value ();
#else
  int nargin = args.length ();

  if (! (nargin ==  2 || nargin == 3) || nargout != 0)
    {
      print_usage ();
      return octave_value ();
    }
  if (! (args(0).is_string () && args(1).is_string ()))
    {
      print_usage ();
      return octave_value ();
    }
  if (nargin == 3 && ! args(2).is_string ())
    {
      print_usage ();
      return octave_value ();
    }
  
  string filename = args(0).string_value ();
  string location = args(1).string_value ();
  if (error_state)
    return octave_value ();

  //open the hdf5 file
  H5File file (filename.c_str (), true, true);
  if (error_state)
    return octave_value ();
  if (nargin == 2)
    file.delete_link (location.c_str ());
  else if (nargin == 3)
    {
      string attname = args(2).string_value ();
      if(!error_state)
	file.delete_att (location.c_str (), attname.c_str ());
    }
  
  return octave_value ();
#endif
}

#if defined (HAVE_HDF5) && defined (HAVE_HDF5_18)

H5File::H5File (const char *filename, const bool create_if_nonexisting,
		const bool write_access)
{
  H5E_auto_t oef;
  void *olderr;
  H5Eget_auto (H5E_DEFAULT,&oef,&olderr);
  //suppress hdf5 error output
  H5Eset_auto (H5E_DEFAULT,0,0);

  file_stat fs (filename);
  if (! fs.exists () && create_if_nonexisting)
    file = H5Fcreate (filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  else if (! fs.exists () && ! create_if_nonexisting)
    error ("The file %s does not exist: %s", filename, strerror (errno));
  else
    {
      // test if the existing file is in HDF5 format
      if (! H5Fis_hdf5 (filename))
        error ("The file is not in the HDF5 format, %s: %s", filename, strerror (errno));
      else
        {
          file = H5Fopen (filename,
			  write_access ? H5F_ACC_RDWR : H5F_ACC_RDONLY,
			  H5P_DEFAULT);
          if (file < 0)
            error ("Opening the file failed, %s: %s", filename, strerror (errno));
        }
    }
  // restore old setting
  H5Eset_auto (H5E_DEFAULT,oef,olderr);
}

H5File::~H5File ()
{
  if (H5Iis_valid (memspace_id))
    H5Sclose (memspace_id);

  if (H5Iis_valid (dspace_id))
    H5Sclose (dspace_id);

  if (H5Iis_valid (dset_id))
    H5Dclose (dset_id);
  
  if (H5Iis_valid (att_id))
    H5Aclose (att_id);

  if (H5Iis_valid (obj_id))
    H5Oclose (obj_id);

  if (H5Iis_valid (type_id))
    H5Tclose (type_id);
  
  if (H5Iis_valid (mem_type_id))
    H5Tclose (mem_type_id);

  if (H5Iis_valid (file))
    H5Fclose (file);


  if (h5_dims != NULL)
    {
      free (h5_dims);
      h5_dims = NULL;
    }
  if (h5_maxdims != NULL)
    {
      free (h5_maxdims);
      h5_maxdims = NULL;
    }
}

// T will be Matrix or dim_vector
template <typename T>
hsize_t*
H5File::alloc_hsize (const T& dim, const int mode, const bool reverse)
{
  int rank = dim.numel ();
  hsize_t *hsize = (hsize_t*)malloc (rank * sizeof (hsize_t));
  for (int i = 0; i < rank; i++)
    {
      int j = reverse ? rank-i-1 : i;
      if (mode == ALLOC_HSIZE_INFZERO_TO_UNLIMITED && (dim(i) == octave_Inf || dim(i) == 0))
        hsize[j] = H5S_UNLIMITED;
      else if (mode == ALLOC_HSIZE_INF_TO_ZERO && dim(i) == octave_Inf)
        hsize[j] = 0;
      else
        hsize[j] = dim(i);
    }
  return hsize;
}


int
H5File::open_dset (const char *dsetname)
{
  dset_id = H5Dopen (file, dsetname, H5P_DEFAULT);
  if (dset_id < 0)
    {
      error ("Error opening dataset %s", dsetname);
      return -1;
    }

  dspace_id = H5Dget_space (dset_id);
  if (dspace_id < 0)
    {
      error ("Error opening dataspace of dataset %s", dsetname);
      return -1;
    }

  rank = H5Sget_simple_extent_ndims (dspace_id);
  if (rank < 0)
    {
      error ("Error reading extent of %s", dsetname);
      return -1;
    }

  h5_dims = (hsize_t*)malloc (rank * sizeof(hsize_t));
  h5_maxdims = (hsize_t*)malloc (rank * sizeof(hsize_t));
  if (! h5_dims || ! h5_maxdims)
    {
      error ("Error allocating memory for %s", dsetname);
      return -1;
    }
  if (H5Sget_simple_extent_dims (dspace_id, h5_dims, h5_maxdims) < 0)
    {
      error ("Error determining current dimensions and maximum size of dataset %s", dsetname);
      return -1;
    }
  
  return 0;
}

octave_value
H5File::read_dset_complete (const char *dsetname)
{
  if (open_dset (dsetname) < 0)
    return octave_value ();

  mat_dims.resize (max (rank, 2));
  // .resize(1) still leaves mat_dims with a length of 2, so
  // we need at least 2 filled
  mat_dims(0) = mat_dims(1) = 1;
  for (int i = 0; i < rank; i++)
    //note that this is reversing the order
    mat_dims(i) = h5_dims[rank-i-1];

  if (H5Sselect_all (dspace_id) < 0)
    {
      error ("Error selecting complete dataset %s", dsetname);
      return octave_value ();
    }

  octave_value retval = read_dset ();
  return retval;
}

octave_value
H5File::read_dset_hyperslab (const char *dsetname,
                             const Matrix& start, const Matrix& count,
                             const Matrix& stride, const Matrix& block,
                             int nargin)
{
  if (open_dset (dsetname) < 0)
    return octave_value ();

  if (rank == 0 && ! (start.is_empty () && count.is_empty ()
                      && stride.is_empty () && block.is_empty ()))
    {
      error ("Cannot specify hyperslab for scalar datasets (rank 0)");
      return octave_value ();
    }

  if (start.numel () != rank)
    {
      error ("start must be a vector of length %d, the dataset rank", rank);
      return octave_value ();
    }
  if (count.numel () != rank)
    {
      error ("count must be a vector of length %d, the dataset rank", rank);
      return octave_value ();
    }
  
  Matrix _stride = stride;
  if (nargin < 3)
    _stride = Matrix (dim_vector(1, rank), 1);
  if (_stride.numel () != rank)
    {
      error ("stride must be a vector of length %d, the dataset rank", rank);
      return octave_value ();
    }
  Matrix _block = block;
  if (nargin < 4)
    _block = Matrix (dim_vector(1, rank), 1);
  if (_block.numel () != rank)
    {
      error ("block must be a vector of length %d, the dataset rank", rank);
      return octave_value ();
    }

  // .resize(1) still leaves mat_dims with a length of 2, so
  // we need at least 2 filled
  mat_dims.resize (max (rank, 2));
  mat_dims(0) = mat_dims(1) = 1;

  Matrix _count = count;
  for (int i = 0; i < rank; i++)
    {
      if (_stride(i) < _block(i))
        {
          error ("In dimension %d, requested stride %d smaller than block size %d",
                 i+1, (int)_stride(i), (int)_block(i));
          return octave_value ();
        }
      if (_count(i) == 0)
        {
          // a value of 0 (or Inf) means that as many blocks as possible
          // shall be read in this dimension
          _count(i) = (h5_dims[rank-i-1] - start(i) - _block(i)) / _stride(i) + 1;
        }
      mat_dims(i) = _count(i)*_block(i);
      int end = start(i) + _stride(i)*(_count(i)-1) + _block(i); // exclusive
      if (h5_dims[rank-i-1] < end)
        {
          error ("In dimension %d, dataset only has %d elements, but at least %d"
                 " are required for requested hyperslab", i+1, (int)h5_dims[rank-i-1],
                 end);
          return octave_value ();
        }
    }

  hsize_t *hstart = alloc_hsize (start, ALLOC_HSIZE_DEFAULT, true);
  if(hstart == NULL)
    {
      error ("error when allocating hstart array, for %s", dsetname);
      return octave_value ();
    }
  hsize_t *hstride = alloc_hsize (_stride, ALLOC_HSIZE_DEFAULT, true);
  if(hstride == NULL)
    {
      error ("error when allocating hstride array, for %s", dsetname);
      return octave_value ();
    }
  hsize_t *hcount = alloc_hsize (_count, ALLOC_HSIZE_DEFAULT, true);
  if(hcount == NULL)
    {
      error ("error when allocating hcount array, for %s", dsetname);
      return octave_value ();
    }
  hsize_t *hblock = alloc_hsize (_block, ALLOC_HSIZE_DEFAULT, true);
  if(hblock == NULL)
    {
      error ("error when allocating hblock array, for %s", dsetname);
      return octave_value ();
    }
  
  herr_t sel_result = H5Sselect_hyperslab (dspace_id, H5S_SELECT_SET, hstart,
                                           hstride, hcount, hblock);
  free (hstart);
  free (hstride);
  free (hcount);
  free (hblock);
  if (sel_result < 0)
    {
      error ("error when selecting the hyperslab of dataset %s to read from", dsetname);
      return octave_value ();
    }
  
  octave_value retval = read_dset ();
  return retval;
  
}

octave_value
H5File::read_dset ()
{
  bool is_cmplx = false;
  type_id = H5Dget_type (dset_id);
  hid_t complex_type_id = hdf5_make_complex_type (H5T_NATIVE_DOUBLE);
  
  hsize_t *hmem = alloc_hsize (mat_dims, ALLOC_HSIZE_DEFAULT, false);
  hid_t memspace_id = H5Screate_simple (rank, hmem, hmem);
  free (hmem);
  if (memspace_id < 0)
    {
      return octave_value ();
    }

  if (H5Sselect_valid (dspace_id) <= 0)
    {
      error ("selected dataspace is not valid");
      return octave_value ();
    }

  octave_value retval;
  herr_t read_result;
  if (H5Tget_class (type_id) == H5T_COMPOUND &&
      H5Tget_class (complex_type_id) == H5T_COMPOUND &&
      hdf5_types_compatible (type_id, complex_type_id) > 0)
    {
      ComplexNDArray ret (mat_dims);
      // macro begin
#define HDF5_READ_DATA(type)                                            \
      read_result = H5Dread (dset_id,                                   \
                             type,                                      \
                             memspace_id, dspace_id,                    \
                             H5P_DEFAULT, ret.fortran_vec ());          \
      if (read_result < 0)                                              \
        {                                                               \
          error ("error when reading dataset");                         \
          return octave_value ();                                  \
        }                                                               \
      retval = octave_value (ret)
      // macro end
      
      HDF5_READ_DATA (type_id);
    }
  else if (H5Tget_class (type_id) == H5T_INTEGER)
    {
      switch (H5Tget_size (type_id)*8)
        {
        case 64:
          if (H5Tget_sign (type_id) == H5T_SGN_NONE)
            {
              uint64NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          else
            {
              int64NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          break;
        case 32:
          if (H5Tget_sign (type_id) == H5T_SGN_NONE)
            {
              uint32NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          else
            {
              int32NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          break;
        case 16:
          if (H5Tget_sign (type_id) == H5T_SGN_NONE)
            {
              uint16NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          else
            {
              int16NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          break;
        case 8:
          if (H5Tget_sign (type_id) == H5T_SGN_NONE)
            {
              uint8NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          else
            {
              int8NDArray ret (mat_dims);
              HDF5_READ_DATA (type_id);
              break;
            }
          break;
        default:
          {
            error ("unknown integer size %d", H5Tget_size (type_id));
            NDArray ret (mat_dims);
            HDF5_READ_DATA (type_id);
          }
        }
    }
  else
    {
      NDArray ret (mat_dims);
      HDF5_READ_DATA (H5T_NATIVE_DOUBLE);
    }
  H5Tclose (complex_type_id);
  
  return retval;
}

void
H5File::write_dset (const char *dsetname,
                    const octave_value ov_data)
{
  int rank = ov_data.dims ().numel ();

  hsize_t *dims = alloc_hsize (ov_data.dims(), ALLOC_HSIZE_DEFAULT, true);
  dspace_id = H5Screate_simple (rank, dims, NULL);
  free (dims);

  // determine the endianness of this system
  H5T_order_t o = H5Tget_order (H5T_NATIVE_INT);
  if (o == H5T_ORDER_ERROR)
    {
      error ("HDF5 lib could not determine endianness of current system");
      return;
    }

  //check if all groups in the path dsetname exist. if not, create them
  string path (dsetname);
  for (int i=1; i < path.length (); i++)
    {
      if (path[i] == '/')
        {
          if (! H5Lexists (file, path.substr(0,i).c_str (), H5P_DEFAULT))
            {
              hid_t group_id = H5Gcreate (file, path.substr(0,i).c_str (), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
              H5Gclose (group_id);
            }
        }
    }

  herr_t status;
  // find the right type
  if (ov_data.is_complex_type())
    {
      //check if the data set already exists. if it does, open it,
      //otherwise, create it.  Furthermore check if the datatype is
      //compliant with given octave data.
  
#define OPEN_AND_WRITE if (H5Lexists (file,dsetname,H5P_DEFAULT))       \
        {                                                               \
          if (open_dset (dsetname) < 0)                                 \
            {                                                           \
              error ("Could not open existing dataset in order to write to"); \
              return;                                                   \
            }                                                           \
        }                                                               \
      else                                                              \
        dset_id = H5Dcreate (file, dsetname, type_id, dspace_id,        \
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);    \
                                                                        \
      status = H5Dwrite (dset_id, type_id,                              \
                         H5S_ALL, H5S_ALL, H5P_DEFAULT,                 \
                         data.fortran_vec ())
  
      type_id = hdf5_make_complex_type (H5T_NATIVE_DOUBLE);
      ComplexNDArray data = ov_data.complex_array_value ();
      OPEN_AND_WRITE;
    }
  else if (ov_data.is_integer_type ())
    {
      if (ov_data.is_uint64_type ())
        {
          uint64NDArray data = ov_data.uint64_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_U64LE : H5T_STD_U64BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_uint32_type ())
        {
          uint32NDArray data = ov_data.uint32_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_U32LE : H5T_STD_U32BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_uint16_type ())
        {
          uint16NDArray data = ov_data.uint16_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_U16LE : H5T_STD_U16BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_uint8_type ())
        {
          uint8NDArray data = ov_data.uint8_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_U8LE : H5T_STD_U8BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_int64_type ())
        {
          int64NDArray data = ov_data.int64_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_I64LE : H5T_STD_I64BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_int32_type ())
        {
          int32NDArray data = ov_data.int32_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_I32LE : H5T_STD_I32BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_int16_type ())
        {
          int16NDArray data = ov_data.int16_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_I16LE : H5T_STD_I16BE);
          OPEN_AND_WRITE;
        }
      else if (ov_data.is_int8_type ())
        {
          int8NDArray data = ov_data.int8_array_value ();
          type_id = H5Tcopy (o == H5T_ORDER_LE ? H5T_STD_I8LE : H5T_STD_I8BE);
          OPEN_AND_WRITE;
        }
      else
        {
          uint64NDArray data = ov_data.uint64_array_value ();
          type_id = H5Tcopy (H5T_NATIVE_INT);
          OPEN_AND_WRITE;
        }
      
    }
  else if (ov_data.is_single_type ())
    {
      FloatNDArray data = ov_data.float_array_value ();
      type_id = H5Tcopy (H5T_NATIVE_FLOAT);
      OPEN_AND_WRITE;
    }
  else
    {
      NDArray data = ov_data.array_value ();
      type_id = H5Tcopy (H5T_NATIVE_DOUBLE);
      OPEN_AND_WRITE;
    }

  if (status < 0)
    {
      error ("error when writing the dataset %s", dsetname);
      return;
    }

}

void
H5File::write_dset_hyperslab (const char *dsetname,
                              const octave_value ov_data,
                              const Matrix& start, const Matrix& count,
                              const Matrix& stride, const Matrix& block,
                              int nargin)
{
  NDArray data = ov_data.array_value ();

  if (open_dset (dsetname) < 0)
    return;

  // check if the given hyperslab settings are reasonable
  if (rank == 0 && ! (start.is_empty () && count.is_empty ()
                      && stride.is_empty () && block.is_empty ()))
    {
      error ("Cannot specify hyperslab for scalar datasets (rank 0)");
      return;
    }

  if (start.numel () != rank)
    {
      error ("start must be a vector of length %d, the dataset rank", rank);
      return;
    }
  if (count.numel () != rank)
    {
      error ("count must be a vector of length %d, the dataset rank", rank);
      return;
    }
  Matrix _stride = stride;
  if (nargin < 3)
    _stride = Matrix (dim_vector(1, rank), 1);
  if (_stride.numel () != rank)
    {
      error ("stride must be a vector of length %d, the dataset rank", rank);
      return;
    }
  Matrix _block = block;
  if (nargin < 4)
    _block = Matrix (dim_vector(1, rank), 1);
  if (_block.numel () != rank)
    {
      error ("block must be a vector of length %d, the dataset rank", rank);
      return;
    }

  // check further for every dimension if hyperslab settings make sense.
  for (int i = 0; i < rank; i++)
    {
      // the stride must be at least the block size
      if (_stride(i) < _block(i))
        {
          error ("In dimension %d, requested stride %d smaller than block size %d",
                 i+1, (int)_stride(i), (int)_block(i));
          return;
        }

      // A count value 0 is not allowed when writing data.

      int end = start(i) + _stride(i)*(count(i)-1) + _block(i); // exclusive
      if (h5_maxdims[rank-i-1] < end)
        {
          error ("In dimension %d, the dataset %s may have at max. only %d elements,"
                 " but at least %d are required for requested hyperslab.",
                 i+1, dsetname, (int)h5_maxdims[rank-i-1], end);
          return;
        }

      // now, the array holding the current dimension of the dataset
      // is changed (if its necessary), so that the new extent can be
      // set later.
      if (h5_dims[rank-i-1] < end)
        h5_dims[rank-i-1] = end;
    }
  hsize_t *hstart = alloc_hsize (start, ALLOC_HSIZE_DEFAULT, true);
  if(hstart == NULL)
    {
      error ("error when allocating hstart array, for %s", dsetname);
      return;
    }
  hsize_t *hstride = alloc_hsize (_stride, ALLOC_HSIZE_DEFAULT, true);
  if(hstride == NULL)
    {
      error ("error when allocating hstride array, for %s", dsetname);
      return;
    }
  hsize_t *hcount = alloc_hsize (count, ALLOC_HSIZE_DEFAULT, true);
  if(hcount == NULL)
    {
      error ("error when allocating hcount array, for %s", dsetname);
      return;
    }
  hsize_t *hblock = alloc_hsize (_block, ALLOC_HSIZE_DEFAULT, true);
  if(hblock == NULL)
    {
      error ("error when allocating hblock array, for %s", dsetname);
      return;
    }
  
  // make the current size of the dataset bigger
  H5Sclose (dspace_id);
  if (H5Dset_extent (dset_id, h5_dims) < 0)
    {
      error ("error when setting new extent of the dataset %s", dsetname);
      return;
    }
  dspace_id = H5Dget_space (dset_id);
  if (dspace_id < 0)
    {
      error ("error could not get dataspace after setting new extent of %s", dsetname);
      return;
    }
  herr_t sel_result = H5Sselect_hyperslab (dspace_id, H5S_SELECT_SET, hstart,
                                           hstride, hcount, hblock);

  free (hstart);
  free (hstride);
  free (hcount);
  free (hblock);

  if (sel_result < 0)
    {
      error ("error when selecting the hyperslab of dataset %s to write to", dsetname);
      return;
    }
  
  hsize_t *hmem = alloc_hsize (data.dims (), ALLOC_HSIZE_DEFAULT, false);
  if(hmem == NULL)
    {
      error ("error when allocating hmem array, for %s", dsetname);
      return;
    }
  hid_t memspace_id = H5Screate_simple (rank, hmem, hmem);
  if (memspace_id < 0)
    {
      error ("error when creating dataspace for data in memory");
      return;
    }
  free (hmem);
  
  herr_t status = H5Dwrite (dset_id, H5T_NATIVE_DOUBLE,
                            memspace_id, dspace_id, H5P_DEFAULT,
                            data.fortran_vec ());
  if (status < 0)
    {
      error ("error when writing the dataset %s", dsetname);
      return;
    }
  
}


octave_value
H5File::read_att (const char *objname, const char *attname)
{
  octave_value retval;
  obj_id = H5Oopen (file, objname, H5P_DEFAULT);

  if (obj_id < 0)
    {
      error ("h5readatt: opening the given object failed");
      return retval;
    }

  if ( !hdf5_check_attr(obj_id, attname))
    {
      error ("h5readatt: the object %s does not have an attribute %s", objname, attname);
      return retval;
    }

  att_id = H5Aopen_name (obj_id, attname);
  if (att_id < 0)
    {
      error ("h5readatt: opening the given attribute failed");
      return retval;
    }

  hid_t type = H5Aget_type (att_id);
  if (type < 0)
    {
      error ("h5readatt: dataset type error");
      return retval;
    }

  size_t numVal = H5Aget_storage_size (att_id)/H5Tget_size (type);
  if (H5Tget_class (type)==H5T_STRING)
    {
      // Size of each string:
      size_t size = H5Tget_size (type);
      // to read an array of strings (for future work): 
      //totsize = size*sdim[0]*sdim[1];
      // to read a single string:
      size_t totsize = size;
      // Set up read buffer for attribute
      char* buf = (char*)calloc (totsize, sizeof (char));
      if (H5Aread (att_id, type, buf)<0)
        {
          error ("h5readatt: reading the given string Attribute failed");
          return retval;
        }
      retval = octave_value (buf);
    }
  else if (H5Tget_class (type)==H5T_INTEGER)
    {
      // Integer attributes are casted to floating point octave values
    
      double value[numVal];
      if (H5Tget_size (type)==sizeof (int))
        {
          int f_value[numVal];
          if (H5Aread (att_id, H5T_NATIVE_INT, f_value)<0)
            {
              error ("h5readatt: reading the given integer Attribute failed");
              return retval;
            }
          for (size_t n=0;n<numVal;++n)
            value[n] = f_value[n]*1.0;
        }
      else
        {
          error ("h5readatt: reading the given integer Attribute failed: cannot handle size of type");
          return retval;
        }

      Matrix mat (numVal,1);
      for (size_t n=0;n<numVal;++n)
        mat(n)=value[n];
      retval = octave_value (mat);
    
    }
  else if (H5Tget_class (type) == H5T_FLOAT)
    {
      double value[numVal];
      if (H5Tget_size (type) == sizeof (float))
        {
          float f_value[numVal];
          if (H5Aread (att_id, H5T_NATIVE_FLOAT, f_value)<0)
            {
              error ("h5readatt: reading the given float Attribute failed");
              return retval;
            }
          for (size_t n = 0; n < numVal; ++n)
            value[n] = f_value[n];
        }
      else if (H5Tget_size (type) == sizeof (double))
        {
          if (H5Aread (att_id, H5T_NATIVE_DOUBLE, value)<0)
            {
              error ("h5readatt: reading the given double Attribute failed");
              return retval;
            }
        }
      else
        {
          error ("h5readatt: reading the given float Attribute failed: \
cannot handle size of type");
          return retval;
        }

      Matrix mat (numVal,1);
      for (size_t n=0;n<numVal;++n)
        mat(n)=value[n];
      retval = octave_value (mat);

    }
  else //none of the supported data types
    {
      error ("h5readatt: attribute type not supported");
      return retval;
    }
  
  return retval;
}

void
H5File::write_att (const char *location, const char *attname,
                   const octave_value& attvalue)
{
  hsize_t *dims;
  if (attvalue.is_scalar_type () || attvalue.is_string ())
    dspace_id = H5Screate (H5S_SCALAR);
  else if (attvalue.is_matrix_type ())
    {
      error ("matrix type attributes are not yet supported.");
      return;
      // dims = alloc_hsize (attvalue.dims (), ALLOC_HSIZE_DEFAULT, true);
      // dspace_id = H5Screate_simple (attvalue.dims ().numel (), dims, NULL);
      // free (dims);
    }
  else
    {
      error ("Only scalar attributes are supported at the moment.");
      return;
    }

  //H5Lexists returns false for the root group '/'
  if (strcmp (location,"/")!=0 && ! H5Lexists (file, location, H5P_DEFAULT))
    {
      error ("the specified HDF5 object %s does not exist", location);
      return;
    }
  obj_id = H5Oopen (file, location, H5P_DEFAULT);
  if (obj_id < 0)
    {
      error ("the specified HDF5 object %s could not be opened", location);
      return;
    }
  
  //Check if an attribute with the given name exists already at that
  //object and if yes delete it.
  htri_t exists = H5Aexists (obj_id, attname);
  if (exists > 0)
    {
      if (H5Adelete (obj_id, attname) < 0)
        {
          error ("could not delete existing attribute %s at %s", attname, location);
          return;
        }
    }
  else if (exists < 0)
    {
      error ("could not check if attribute %s exists at %s", attname, location);
      return;
    }

  void* buf;
  double attval_double;
  int attval_int;

  if (attvalue.is_string ())
    {
      type_id = H5Tcopy (H5T_C_S1);
      H5Tset_size (type_id, attvalue.string_value ().length ());
      H5Tset_strpad (type_id,H5T_STR_NULLTERM);
      mem_type_id = H5Tcopy (type_id);
      
      buf = (void *) attvalue.string_value ().c_str ();
    }
  else if (attvalue.is_integer_type ())
    {
      //type_id = H5Tcopy (H5T_STD_I64LE); //cannot read this back in then, don't know why
      type_id = H5Tcopy (H5T_NATIVE_INT);
      mem_type_id = H5Tcopy (H5T_NATIVE_INT);
      attval_int = attvalue.int_value ();
      buf = (void *) &attval_int;
    }
  else if (attvalue.is_real_type ())
    {
      type_id = H5Tcopy (H5T_NATIVE_DOUBLE);
      mem_type_id = H5Tcopy (H5T_NATIVE_DOUBLE);
      attval_double = attvalue.double_value ();
      buf = (void *) &attval_double;
    }
  else if (attvalue.is_complex_type ())
    {
      error ("complex values are not supported by the HDF5 format. \
You have to save real and imag part separately.");
      return;
    }
  else
    {
      error ("this variable type is not supported");
      return;
    }

  if (H5Aexists (obj_id,attname))
    att_id = H5Aopen (obj_id, attname, H5P_DEFAULT);
  else
    {
      att_id = H5Acreate (obj_id, attname, type_id,
                          dspace_id, H5P_DEFAULT, H5P_DEFAULT);
    }
  herr_t status = H5Awrite (att_id, mem_type_id, buf);
  if (status < 0)
    {
      error ("error when writing the attribute %s at %s", attname, location);
      return;
    }
}


void
H5File::create_dset (const char *location, const Matrix& size,
                     const char *datatype, Matrix& chunksize)
{
  int typesize;
  if (strcmp (datatype,"double") == 0)
    {
      type_id = H5Tcopy (H5T_NATIVE_DOUBLE);
      typesize = sizeof(double);
    }
  else if (strcmp (datatype,"single") == 0)
    {
      type_id = H5Tcopy (H5T_NATIVE_FLOAT);
      typesize = sizeof(float);
    }
  else if (strcmp (datatype,"uint64") == 0)
    {
      type_id = H5Tcopy (H5T_STD_U64LE);
      typesize = 64/8;
    }
  else if (strcmp (datatype,"uint32") == 0)
    {
      type_id = H5Tcopy (H5T_STD_U32LE);
      typesize = 32/8;
    }
  else if (strcmp (datatype,"uint16") == 0)
    {
      type_id = H5Tcopy (H5T_STD_U16LE);
      typesize = 16/8;
    }
  else if (strcmp (datatype,"uint8") == 0)
    {
      type_id = H5Tcopy (H5T_STD_U8LE);
      typesize = 8/8;
    }
  else if (strcmp (datatype,"int64") == 0)
    {
      type_id = H5Tcopy (H5T_STD_I64LE);
      typesize = 64/8;
    }
  else if (strcmp (datatype,"int32") == 0)
    {
      type_id = H5Tcopy (H5T_STD_I32LE);
      typesize = 32/8;
    }
  else if (strcmp (datatype,"int16") == 0)
    {
      type_id = H5Tcopy (H5T_STD_I16LE);
      typesize = 16/8;
    }
  else if (strcmp (datatype,"int8") == 0)
    {
      type_id = H5Tcopy (H5T_STD_I8LE);
      typesize = 8/8;
    }
  else
    {
      error ("invalid datatype %s for dataset %s",datatype,location);
      return;
    }

  // the size array may contains Infs, which are casted to zeros for..
  hsize_t *dims = alloc_hsize (size, ALLOC_HSIZE_INF_TO_ZERO, true);
  // and produce unlimited maximum extent for..
  hsize_t *maxdims = alloc_hsize (size, ALLOC_HSIZE_INFZERO_TO_UNLIMITED, true);
  dspace_id = H5Screate_simple (size.numel (), dims, maxdims);
  free (dims);
  free (maxdims);

  if (any_int_leq_zero (size) && chunksize.is_empty())
    {
      error ("If the size argument contains an Inf or zero element, then ChunkSize must be specified.");
      return;
    }
  // get a dataset creation property list
  hid_t crp_list = H5Pcreate (H5P_DATASET_CREATE);
  if (! chunksize.is_empty())
    {
      // a dataset with an unlimited dimension must be chunked.
      if (chunksize(0) == 0)
	chunksize = get_auto_chunksize(size, typesize);
      
      hsize_t *dims_chunk = alloc_hsize (chunksize, ALLOC_HSIZE_DEFAULT, true);
      if (H5Pset_layout (crp_list, H5D_CHUNKED) < 0)
        {
          error ("Could not set chunked layout of %s", location);
          return;
        }
      if (H5Pset_chunk (crp_list, size.numel (), dims_chunk) < 0)
        {
          error ("Could not set chunk size of %s", location);
          return;
        }
      free (dims_chunk);
    }

  // create non-existent intermediate groups.
  hid_t lcpl_list = H5Pcreate (H5P_LINK_CREATE);
  H5Pset_create_intermediate_group (lcpl_list, 1);
  dset_id = H5Dcreate (file, location, type_id, dspace_id,
                       lcpl_list, crp_list, H5P_DEFAULT);
  if (dset_id < 0)
    {
      error ("Could not create dataset %s", location);
      return;
    }
  H5Pclose (crp_list);
  H5Pclose (lcpl_list);

}

void
H5File::delete_link (const char *location)
{
  herr_t status = H5Ldelete (file, location, H5P_DEFAULT);
  if (status < 0)
    {
      error ("Error when deleting object %s", location);
      return;
    }
}


void
H5File::delete_att (const char *location, const char *att_name)
{
  herr_t status = H5Adelete_by_name (file,location,att_name,H5P_DEFAULT);
  if (status < 0)
    {
      error ("Error when deleting attribute %s of object %s", att_name, location);
      return;
    }
}


Matrix
H5File::get_auto_chunksize(const Matrix& dset_shape, int typesize)
{
  // This function originally stems from the h5py project.
  
  // Guess an appropriate chunk layout for a dataset, given its shape and
  // the size of each element in bytes. Will allocate chunks only as large
  // as MAX_SIZE. Chunks are generally close to some power-of-2 fraction of
  // each axis, slightly favoring bigger values for the last index.
  const int CHUNK_BASE = 16*1024; // Multiplier by which chunks are adjusted
  const int CHUNK_MIN = 8*1024;  //Soft lower limit (8k)
  const int CHUNK_MAX = 1024*1024; // Hard upper limit (1M)

  Matrix chunksize = dset_shape;
  int ndims = chunksize.numel ();
  for (int i = 0; i < ndims; i++)
    {
      //For unlimited dimensions we have to guess 1024
      if(chunksize(i) == octave_Inf || chunksize(i) == 0)
	chunksize(i) = 1024;
    }
  // Determine the optimal chunk size in bytes using a PyTables expression.
  // This is kept as a float.
  int dset_size = chunksize.prod ()(0)*typesize;
  int target_size = CHUNK_BASE * pow(2,log10(dset_size/(1024.0 * 1024)));
  if (target_size > CHUNK_MAX)
    target_size = CHUNK_MAX;
  else if (target_size < CHUNK_MIN)
    target_size = CHUNK_MIN;

  int idx = 0;
  while(true)
    {

      // Repeatedly loop over the axes, dividing them by 2. Stop when:
      // 1a. We're smaller than the target chunk size, OR
      // 1b. We're within 50% of the target chunk size, AND
      // 2. The chunk is smaller than the maximum chunk size
      int chunk_bytes = chunksize.prod ()(0)*typesize;
      if ((chunk_bytes < target_size ||
	   abs(chunk_bytes-target_size)/target_size < 0.5) &&
	  chunk_bytes < CHUNK_MAX)
	break;
      
      if (chunksize.prod ()(0) == 1)
	break; // Element size larger than CHUNK_MAX
      
      chunksize(idx%ndims) = ceil(chunksize(idx%ndims) / 2.0);
      idx++;
    }
  return chunksize;
}


#endif
