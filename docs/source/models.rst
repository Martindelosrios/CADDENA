.. _Models:

Available Models
----------------

#. XENONnT_O1_rate: 

   This model was trained using synthetic data of the expected total
   rate in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in [6-1000] GeV`, :math:`\sigma \in [10^{-50}-10^{-43}] cm^{2}` 
   and :math:`\theta \in [-\pi/2-\pi/2]`.
   In order to analyse a new data :math:`x` it must be a np.array with shape (1). CHECK THIS

#. XENONnT_O1_drate: 

   This model was trained using synthetic data of the expected differential
   rate in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in [6-1000] GeV`, :math:`\sigma \in [10^{-50}-10^{-43}] cm^{2}]` 
   and :math:`\theta \in [-\pi/2-\pi/2]`.
   The differential rate is just the number of events in recoil energy bins. 
   We have considered a bin width of :math:`1keV$_{NR}$` and a range :math:`(3, 61) keV$_{NR}$`.
   In order to analyse a new data :math:`x` it must be a np.array with shape (n,59), where n is the number of observations to be analysed. CHECK THIS

#. XENONnT_O1_s1s2: 

   This model was trained using synthetic data of the expected s1s2
   data in xenonNT for dark matter particles with :math:`\mathcal{O}_{1}`
   with :math:`M_{dm} \in [6-1000] GeV`, :math:`\sigma \in [10^{-50}-10^{-43}] cm^{2}]` 
   and :math:`\theta \in [-\pi/2-\pi/2]`.
   In cS1 we considered a range of (3, 100) PE with 97 bins, and 70 bins in cS2 within (100, 10000) PE.
   In order to analyse a new data :math:`x` it must be a np.array with shape (n,97,70), where n is the number of observations to be analysed. CHECK THIS
