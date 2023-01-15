#include "crossover.h"
#include <iostream>

using Client = int;
using Route = std::vector<Client>;
using Routes = std::vector<Route>;

namespace
{
struct InsertPos  // best insert position, used to plan unplanned clients
{
    int deltaCost;
    Route *route;
    size_t offset;
};

// Evaluates the cost change of inserting client between prev and next.
int deltaCost(Client client, Client prev, Client next, Params const &params)
{
    int prevEarliestArrival
        = std::max(params.dist(0, prev), params.clients[prev].twEarly);
    int prevEarliestFinish = prevEarliestArrival + params.clients[prev].servDur;
    int distPrevClient = params.dist(prev, client);
    int clientLate = params.clients[client].twLate;

    if (prevEarliestFinish + distPrevClient >= clientLate)
        return INT_MAX;

    int clientEarliestArrival
        = std::max(params.dist(0, client), params.clients[client].twEarly);
    int clientEarliestFinish
        = clientEarliestArrival + params.clients[client].servDur;
    int distClientNext = params.dist(client, next);
    int nextLate = params.clients[next].twLate;

    if (clientEarliestFinish + distClientNext >= nextLate)
        return INT_MAX;

    return distPrevClient + distClientNext - params.dist(prev, next);
}
}  // namespace

void crossover::greedyRepair(Routes &routes,
                             std::vector<Client> const &unplanned,
                             Params const &params)
{
    size_t numRoutes = routes.size();

    for (Client client : unplanned)
    {
        InsertPos best = {INT_MAX, &routes.front(), 0};

        for (size_t rIdx = 0; rIdx != numRoutes; ++rIdx)
        {
            auto &route = routes[rIdx];

            // Compute dispatch window of route. Skip this route if its
            // not compatible with the to be inserted client.
            int lastRelease = 0;
            int latestDispatch = INT_MAX;
            for (auto const cust : route)
            {
                lastRelease
                    = std::max(lastRelease, params.clients[cust].releaseTime);
                latestDispatch = std::min(latestDispatch,
                                          params.clients[cust].latestDispatch);
            }

            if (params.clients[client].releaseTime > latestDispatch
                || params.clients[client].latestDispatch < lastRelease)
                continue;

            for (size_t idx = 0; idx <= route.size(); ++idx)
            {
                Client prev, next;

                if (route.empty())  // try empty route
                {
                    prev = 0;
                    next = 0;
                }
                else if (idx == 0)  // try after depot
                {
                    prev = 0;
                    next = route[0];
                }
                else if (idx == route.size())  // try before depot
                {
                    prev = route.back();
                    next = 0;
                }
                else  // try between [idx - 1] and [idx]
                {
                    prev = route[idx - 1];
                    next = route[idx];
                }

                auto const cost = deltaCost(client, prev, next, params);
                if (cost < best.deltaCost)
                    best = {cost, &route, idx};
            }
        }

        auto const [cost, route, offset] = best;
        if (cost == INT_MAX)
        {
            std::cout << "Problem! " << offset << '\n';

            // routes[std::min(numRoutes, routes.size() - 1)].insert(
            //     route->begin(), client);
        }
        // else
        route->insert(route->begin() + static_cast<long>(offset), client);
    }
}
