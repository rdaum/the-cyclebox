# the cyclebox

### STATUS: stalled, until I get time to come back to this and fix a number of things.

  * Finish writing the benchmarks and correctness tests I was working on 
  * Completely rework the use of atomics (and locks) throughout this. 
  
### what am i?

  * A simple lightweight reference counting concurrent garbage collector implementation that 
handles cycles in the reference graph and user-defined memory structure.
  * Very much an early experiment. Not the author's day job. Broken out of a larger hobby project of the author's.
  * Based on the algorithm(s) described in https://pages.cs.wisc.edu/~cymen/misc/interests/Bacon01Concurrent.pdf
  * Implementation follows the pseudocode in the paper quite closely.
  * Written in Rust, but _not_ designed for use as a generalized replacement for `Arc`/`Rc`. 
    e.g. Does not have a "smart pointer", and is not meant for the management of more "general" Rust objects. 
    If you want that, maybe take a look at https://github.com/artichoke/cactusref 
  * Meant _more_ to be used as a general re-usable embeddable library for systems that need GC, such as a language
    interpreter or virtual machine.
  * In an attempt to be more cache-friendly, reference management is handled external to objects themselves.
  * Likewise, object relationships are also declared external to the objects, in an attempt to make the system more
    generic, and to make traces more cache-friendly

### why reference counting?

a few of my thoughts...

  * _simpler to implement / small footprint of implementation_
  * most objects die young, so let's kill them right away
  * in theory possibility of fewer pauses, as long as cycles are shallow
  * with the right implementation, 
  * for _persistent systems_ (where objects are automatically mapped to secondary storage) -- an area of interest 
    to me -- the ability to manage objects across the persistent/transient boundary without expensive (full) traces
    across disk. means we can present a _unified garbage collected persistent heap_ to the application 
  
### current status

  * very early stages, and under development
  * has not been used in a real production system yet
  * some basic benchmark and tests exist, but could definitely be extended.
  * i am a hobby project... (the author gets $paid$ to work on much more in depth and larger things than this.)
  * use at own risk
  * contributions more than welcome
  * feel free to contribute
  * in particular very much untested and my current efforts will be on writing a generalized set of unit tests and
    benchmarking and profiling
  

**author**: _ryan.daum @ gmail.com_
