# Dala Foods LLP: 2025 GHG inventory (summary for Bank Alatau)
*Synthetic case data. GHG Protocol Corporate Standard, operational control; Scope 2 location-based. All 1952 lines of the 2025 purchase ledger: 1269 train lines use organiser labels, 683 test lines use this pipeline's predictions (out/audit_test.csv). Ledger is one year split in two.*

**Total 199,638 tCO2e** (range 152,354-246,923, +/-23.7%). Scope 1 14,088 | Scope 2 (location) 15,993 | Scope 3 169,557. Intensity **1.901 tCO2e per t of product** (105,000 t), 1.664 per t raw milk; Scope 1+2 only 0.2865 t/t.

| Scope | Category | tCO2e | Share | Lines (act/spend) | DQ |
|---|---|---:|---:|---|---:|
| Scope 1 | S1_STATIONARY | 8,530.3 | 4.27% | 12 (12/0) | 1.16 |
| Scope 1 | S1_MOBILE | 4,155.4 | 2.08% | 226 (226/0) | 1.01 |
| Scope 1 | S1_FUGITIVE | 1,402.3 | 0.7% | 13 (13/0) | 1.17 |
| Scope 2 | S2_ELECTRICITY | 14,495.8 | 7.26% | 24 (24/0) | 1.16 |
| Scope 2 | S2_HEAT | 1,497.0 | 0.75% | 7 (7/0) | 1.16 |
| Scope 3 | S3_C1_GOODS | 168,213.5 | 84.26% | 1307 (882/425) | 3.0 |
| Scope 3 | S3_C2_CAPITAL | 776.9 | 0.39% | 8 (0/8) | 4.0 |
| Scope 3 | S3_C5_WASTE | 345.0 | 0.17% | 24 (24/0) | 2.0 |
| Scope 3 | S3_C4_TRANSPORT | 173.6 | 0.09% | 121 (91/30) | 2.18 |
| Scope 3 | S3_C6_TRAVEL | 48.3 | 0.02% | 135 (120/15) | 2.03 |
| TOTAL |  | 199,638.2 | 100.0% | 1877 (1399/478) | 2.72 |

**Activity vs spend:** 99.5% of tCO2e and 74.5% of lines are activity-based; spend-based (net of VAT) 0.5% of tCO2e. Market-based Scope 2: not determinable (no supplier-specific factors, energy attribute certificates or residual mix in the data).

**Hotspots (by factor):** 1. Raw cow milk, farm gate 157,499 t (78.9%); 2. Grid electricity, Kazakhstan (location-based) 14,496 t (7.3%); 3. Natural gas, combustion 8,530 t (4.3%); 4. Diesel fuel, combustion 3,719 t (1.9%); 5. Corrugated board / paperboard 3,374 t (1.7%); 6. Aluminium foil lids 2,993 t (1.5%); 7. PET (preforms, bottles) 1,761 t (0.9%); 8. Purchased heat, Almaty district heating 1,497 t (0.7%); 9. Refrigerant R-404A released 1,300 t (0.7%); 10. Aseptic composite carton 773 t (0.4%)

**Data quality** (1 best..5, |t|-weighted): **2.72**. DQ1 +/-5%: 278 lines, 25,940 t; DQ2 +/-10%: 251 lines, 4,781 t; DQ3 +/-30%: 870 lines, 167,853 t; DQ4 +/-50%: 474 lines, 1,062 t; DQ5 +/-100%: 4 lines, 2 t.

**Uncertainty method:** Assumed ranges, not a statistical result. Informed by IPCC 2006 Guidelines Vol.1 Ch.3 (Uncertainties; metered commercial fuel/energy activity data within a few %), IPCC 2006 Vol.2 Ch.2 (CO2 factors for fuels +/-5% or better), GHG Protocol Scope 3 Standard Ch.7 and its Quantitative Uncertainty Guidance (secondary/spend-based EEIO data: +/-50% or more). Lines sharing one factor are treated as fully correlated (their uncertainties add), different factors as independent (root-sum-square).

**Scope 3 materiality** (material = >=1% of total): C1 Purchased goods & services 84.26% (material); C2 Capital goods 0.39%; C4 Upstream transport 0.09%; C5 Waste 0.17%; C6 Business travel 0.02%. **Not covered:** C3 Fuel- and energy-related (well-to-tank, grid T&D losses): likely material (typically 10-25% of S1+S2); no WTT/T&D factor in emission_factors.csv; C7 Employee commuting (~850 staff): not in ledger (no purchase); survey needed.

**Excluded (0 t):** per diem 20 lines (KZT 0.4 m); tax 17 lines (KZT 70.5 m); duplicate 15 lines (KZT 291.5 m); social 10 lines (KZT 38.1 m); fine 7 lines (KZT 19.3 m); advance 6 lines (KZT 32.5 m). Advances (1610) are excluded because the delivery documents that close them are in the ledger.

**Completeness checks:** raw milk in ledger 121,153 t vs ~120,000 t in profile (101.0%); own-fleet diesel 1,485,149 l = 33,003 l per truck (45 trucks); refrigerant lines 13; flights: all legs; credit notes 10 lines (-21.42 t).

**Traceability:** every number rolls up from out/inventory_lines.csv (line_id, factor, quantity, method, DQ, reason) to the ledger document; test-line reasons in out/audit_test.csv.

**Inbox asks covered:** bank (scopes, traceability, activity/spend, DQ), retailer (intensity per kg/l), CFO (one command, no paid API), IT (masked data only), chief accountant (duplicates, advances), sustainability officer (refrigerants in, flights all legs).
