# Daily solar irradiance records used for the replay

Source: NASA POWER Daily API v2.10.5 (power.larc.nasa.gov), parameter ALLSKY_SFC_SW_DWN
(All Sky Surface Shortwave Downward Irradiance, kW-hr/m^2/day), community RE, source SYN1DEG (CERES SYN1deg),
time standard LST, 1 January 2021 to 31 December 2025 (1,826 days), no missing values.
Retrieved 25 November 2026.

| file | place | latitude | longitude |
|---|---|---|---|
| dhaka_2021_2025.digits  | Dhaka, Bangladesh | 23.81 | 90.41 |
| ibadan_2021_2025.digits | Ibadan, Nigeria   |  7.38 |  3.95 |
| niamey_2021_2025.digits | Niamey, Niger     | 13.51 |  2.11 |

Format: three digits per day, the daily value in hundredths of kWh/m^2/day (rounded to the nearest hundredth),
16 days per line, in date order. SHA-256 of the digits without line breaks:

- dhaka  22450b8ab0645087427f00484cfe11ab5744c59a0570f9ea9b31beca2ae821e8
- ibadan a153bab1ce7b1816cba2f5724b8e17f6d1591af29c551f8ac492d38a6a7d65bf
- niamey 8455c82c2f188a8d63022ad1b2724df515d9cd8bf02c90283a1b67cccd0a3c64

The hashes were computed in the browser on the API's answer and checked again on these files.
