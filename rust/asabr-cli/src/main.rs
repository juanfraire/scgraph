//! Level-0 A-SABR CLI for scgraph.
//!
//! Usage: asabr-cli <cp_file> <source> <dest> <size> <t0> [expiration]
//!
//! Parses an A-SABR-format contact plan, routes one bundle with the SPSN
//! mainframe (SABR distance, hybrid parenting), and prints the best route to
//! <dest> as one JSON object on stdout:
//!   {"found": true, "arrival": <f64>, "hops": [{"to_node": u16, "at_time": f64}, ...]}
//! Hops are listed source-side first; the last hop's at_time is the arrival.

use std::alloc::System;

#[global_allocator]
static GLOBAL: System = System;

use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::exit;
use std::rc::Rc;

use a_sabr::contact_plan::{ContactPlan, asabr_file_lexer};
use a_sabr::parsing::CMDynStandard;
use a_sabr::route_stage::RouteStage;
use a_sabr::{
    bundle::Bundle,
    contact_manager::ContactManager,
    errors::ASABRError,
    node_manager::{NodeManager, none::NoManagement},
    route_storage::cache::TreeCache,
    routing::{Router, aliases::SpsnHybridParenting},
};

fn backtrace<NM: NodeManager, CM: ContactManager>(
    route: &RouteStage<NM, CM>,
    hops: &mut Vec<(u16, f64)>,
) {
    if let Some(via) = &route.via {
        backtrace(&*via.parent_route.borrow(), hops);
    }
    hops.push((route.to_node, route.at_time));
}

fn main() -> Result<(), ASABRError> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "Usage: {} <cp_file> <source> <dest> <size> <t0> [expiration]",
            args[0]
        );
        exit(1);
    }
    let source: u16 = args[2].parse().expect("bad source");
    let dest: u16 = args[3].parse().expect("bad dest");
    let size: f64 = args[4].parse().expect("bad size");
    let t0: f64 = args[5].parse().expect("bad t0");
    let expiration: f64 = if args.len() > 6 {
        args[6].parse().expect("bad expiration")
    } else {
        f64::MAX
    };

    let file = File::open(&args[1]).expect("cannot open contact plan");
    let contact_plan: ContactPlan<NoManagement, CMDynStandard> =
        asabr_file_lexer::parse_from_iter(BufReader::new(file).lines().map(|r| {
            r.map_err(|e| {
                eprintln!("Error while reading file: {e}");
                exit(-1)
            })
            .unwrap()
        }))?;

    let table = Rc::new(RefCell::new(TreeCache::new(true, false, 10)));
    let mut router = SpsnHybridParenting::new(contact_plan, table, false)?;

    let bundle = Bundle {
        source,
        destinations: vec![dest],
        priority: 0,
        size,
        expiration,
    };

    let out = router.route(source, &bundle, t0, &Vec::new());

    let mut best: Option<Vec<(u16, f64)>> = None;
    if let Ok(Some(out)) = out {
        for (_contact, dest_routes) in out.first_hops.values() {
            for route_rc in dest_routes {
                let route = route_rc.borrow();
                if route.to_node == dest {
                    let mut hops = Vec::new();
                    backtrace(&route, &mut hops);
                    let better = match &best {
                        None => true,
                        Some(b) => route.at_time < b.last().unwrap().1,
                    };
                    if better {
                        best = Some(hops);
                    }
                }
            }
        }
    }

    match best {
        Some(hops) => {
            let arrival = hops.last().unwrap().1;
            let hops_json: Vec<serde_json::Value> = hops
                .iter()
                .map(|(n, t)| serde_json::json!({"to_node": n, "at_time": t}))
                .collect();
            println!(
                "{}",
                serde_json::json!({"found": true, "arrival": arrival, "hops": hops_json})
            );
        }
        None => {
            println!("{}", serde_json::json!({"found": false}));
        }
    }
    Ok(())
}
