#pragma once

/*!
  Following debug switches can be used to help troubleshooting and narrow down bugs.
  These represent common debug scenarios and puts them all in a central place.
 */

//------------------------------------------------------------------------------------
// View operations switches
//------------------------------------------------------------------------------------

//
// Enable this switch to get verbose information about the view ops. This includes the graph creation in `chainViewOperation`, and
// all the gather - scatter operations, such as key misses or other corner cases.
//
#define MPS_DEBUG_VIEW_OPS              0