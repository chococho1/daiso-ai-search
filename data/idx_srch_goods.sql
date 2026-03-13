select tpmp.pd_no
	, tpmp.exh_pd_nm
	, tpmp.pd_prc
	, tpmp.new_pd_yn
	, tpmp.brnd_cd ||'>'|| tpmb.brnd_nm as group_brand
	, CONCAT(
		case when tpmp.new_pd_yn = 'Y' then '01>신상품,' end,
		case when tpmp.pkup_or_psbl_yn = 'N' AND tpmp.mass_or_psbl_yn='N' and tpmp.fdrm_or_psbl_yn='N' and tpmp.pds_or_psbl_yn = 'Y' then '02>택배전용,' end,
		case when tpmp.pkup_or_psbl_yn = 'N' and tpmp.pds_or_psbl_yn = 'N' and tpmp.mass_or_psbl_yn='N' and tpmp.fdrm_or_psbl_yn ='N' and tpmqp.pd_no is not null then '03>오늘배송전용,' end,
		case when tpmp.pkup_or_psbl_yn = 'Y' AND tpmp.mass_or_psbl_yn='N' and tpmp.fdrm_or_psbl_yn='N' and tpmp.pds_or_psbl_yn = 'N' and tpmqp.pd_no is null then '04>매장픽업전용,' end,
		case when tpmp.pd_ty = '03' then '05>모바일상품권,' end,
		case when tpmp.pd_ty = '04' then '06>굿즈,' else '' end) as GROUP_PD_TYPE
	, CONCAT(case when tpmp.pds_or_psbl_yn = 'Y' then '01>택배배송,' end,
        case when tpmp.pkup_or_psbl_yn = 'Y' then '02>매장픽업,' end,
		case when tpmqp.pd_no = tpmp.pd_no then '03>오늘배송,' end, -- (2025.01.07 추가)
        case when tpmp.mass_or_psbl_yn = 'Y' then '04>대량배송,' end,
	--   case when tpmp.fdrm_or_psbl_yn = 'Y' then '05>정기배송,' end
        )  AS GROUP_TAKBAE_TYPE
    , tpmp.pkup_or_psbl_yn
	, tpmp.pds_or_psbl_yn
	, tpmp.mass_or_psbl_yn
	, tpmp.fdrm_or_psbl_yn
	, case when tpmqp.pd_no = tpmp.pd_no then 'Y' else 'N' end as quick_or_psbl_yn -- 퀵배송 가능여부 (2025.01.07 추가)
	, case when tpmp.pkup_or_psbl_yn = 'N' and tpmp.pds_or_psbl_yn = 'N' and tpmp.mass_or_psbl_yn='N' and tpmp.fdrm_or_psbl_yn ='N' and tpmqp.pd_no is not null
			then 'Y' else 'N' end as only_quick_yn -- 오늘배송전용어부
	, tsmp.avg_stsc_val
    , tsmp.revw_cnt
    , tsmp.pdimgurl
    , tsmp.only_onl_yn
    , coalesce(tpmos.onl_stck_qy, 0) as onl_stck_qy
    , case when tpmp.pds_or_psbl_yn = 'Y' and coalesce(tpmos.onl_stck_qy, 0) < 1 then 'Y' else 'N' end as soldoutyn
	, tpmp.inct
    from (select *
	from dis01.tb_pd_m_pd
	where sle_sts_cd in ('02','03') and exh_yn = 'Y') tpmp
left outer join dis01.tb_pd_m_brnd tpmb
	on tpmp.brnd_cd = tpmb.brnd_cd
left outer join dis01.tb_pd_m_quick_pd tpmqp -- 오늘배송 관련 테이블
    on tpmp.pd_no = tpmqp.pd_no
left outer join dis01.tb_search_m_pd tsmp
	on tpmp.pd_no = tsmp.pd_no
left outer join  (  select pd_no
     				, sum( case when (coalesce (stck_qy,0) - coalesce (safe_stck_qy,0) - coalesce (dlvy_parm_stck_qy,0)) > 0
                 			then coalesce (stck_qy,0) - coalesce (safe_stck_qy,0) - coalesce (dlvy_parm_stck_qy,0) else 0 end ) as onl_stck_qy
				     from dis01.tb_pd_m_onl_stck	-- 일반 온라인재고수량
				     where use_yn='Y' and del_yn='N'
				     group by pd_no
				    union ALL
				    select  tpmgm.grp_pd_no
				        , sum( case when (coalesce (stck_qy,0) - coalesce (safe_stck_qy,0) - coalesce (dlvy_parm_stck_qy,0)) > 0
				             then coalesce (stck_qy,0) - coalesce (safe_stck_qy,0) - coalesce (dlvy_parm_stck_qy,0) else 0 end ) as onl_stck_qy
				     from dis01.tb_pd_m_grppd_mapp tpmgm left outer join dis01.tb_pd_m_onl_stck tpmos  -- 온라인전용상품의 온라인 재고수량
     on tpmgm.pd_no = tpmos.pd_no
     where use_yn='Y' and del_yn='N'
     group by tpmgm.grp_pd_no) tpmos
    ON tsmp.pd_no  = tpmos.pd_no
 left outer join (  select tpmecm.pd_no
      , c1.exh_ctgr_nm as c1_nm, c1.exh_ctgr_no as c1_no, c1.exh_ctgr_lvl_no  c1_lvl
      , c2.exh_ctgr_nm as c2_nm, c2.exh_ctgr_no as c2_no , c2.exh_ctgr_lvl_no  c2_lvl
      , c3.exh_ctgr_nm as c3_nm, c3.exh_ctgr_no as c3_no , c3.exh_ctgr_lvl_no  c3_lvl
      , string_agg(DISTINCT c1.exh_ctgr_no||'>'||c1.exh_ctgr_nm, '||') as GROUP_LARGE_CTGR
	  , string_agg(DISTINCT c1.exh_ctgr_no||'>'||c2.exh_ctgr_no||'>'||c2.exh_ctgr_nm, '||') as GROUP_MIDDLE_CTGR
	  , string_agg(DISTINCT c1.exh_ctgr_no||'>'||c2.exh_ctgr_no||'>'||c3.exh_ctgr_no||'>'||c3.exh_ctgr_nm,'||') as GROUP_SMALL_CTGR
      from dis01.tb_ds_m_exh_ctgr c1
      inner join dis01.tb_ds_m_exh_ctgr c2
      on c1.exh_ctgr_no = c2.upr_exh_ctgr_no
      inner join dis01.tb_ds_m_exh_ctgr c3
      on c2.exh_ctgr_no = c3.upr_exh_ctgr_no
      inner join dis01.tb_pd_m_exh_ctgr_mpp tpmecm
      on c3.exh_ctgr_no = tpmecm.exh_ctgr_no
      where tpmecm.del_yn = 'N' /* 2231129 추가 */
      and tpmecm.use_yn = 'Y' /* 2231129 추가 */
      and tpmecm.bass_ctgr_yn = 'Y'
      group by tpmecm.pd_no, c1.exh_ctgr_nm, c1.exh_ctgr_no, c2.exh_ctgr_nm, c2.exh_ctgr_no, c3.exh_ctgr_no, c3.exh_ctgr_nm
      ) ctgr_info
    on tsmp.pd_no = ctgr_info.pd_no;
