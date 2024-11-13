0. Analyze your results, when does it make sense to use the various approaches?
 - The GPU approaches up to the size I ran were not super distinguishable in terms of time. cuBLAS seemed consistently a slight margin slower than the manual interpretations (presumably due to API overhead, as the method was likely designed for use with much more complicated operations on matmul variations). Tiled GPU started out a touch faster than untiled, but the difference was marginal at best, and didn't seem to scale comparatively to the algorithms themselves. CPU, of course, was miserable to use at any and all sizes.
How did your speed compare with cuBLAS?
 - We solidly beat cuBLAS, as mentioned above. Also, cuBLAS is hell. 0/10, would not recommend.
What went well with this assignment?
 - Everything other than cuBLAS was super smooth
What was difficult?
 - IN ORDER TO LOAD THE CUBLAS DOCS I HAD TO DOWNLOAD THE WHOLE SITE TO MY DESKTOP AND OPEN THE LOCAL COPY AND I'M STILL MAD ABOUT IT. Also, using the cuBLAS library was hell to begin with, the documentation was frequently vague and I don't know why we were suggested to use GemmBatched methods, as we weren't performing batched multiplication approaches.
How would you approach differently?
 - Probably could have asked someone else for help with the cuBLAS library integration. Other than that, though, I kinda just sped through the whole assignment without many hiccups. Oh, my graph was a bit of a mess because I put it together in half an hour, but I just needed to be done w the assignment so whatever.
Anything else you want me to know?
 - nope