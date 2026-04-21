from __future__ import annotations


TEAM_NAME_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

VENUE_NAME_ALIASES = {
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
}

SCHEDULE_VENUE_TO_GROUND = {
    "Bengaluru": "M Chinnaswamy Stadium",
    "Mumbai": "Wankhede Stadium",
    "Guwahati": "Barsapara Cricket Stadium",
    "New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Delhi": "Arun Jaitley Stadium",
    "Ahmedabad": "Narendra Modi Stadium",
    "Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
    "Chennai": "MA Chidambaram Stadium, Chepauk",
    "Kolkata": "Eden Gardens",
    "Jaipur": "Sawai Mansingh Stadium",
    "Raipur": "Shaheed Veer Narayan Singh International Stadium",
    "Dharamshala": "Himachal Pradesh Cricket Association Stadium",
}


def normalize_team_name(team: str | None) -> str | None:
    if team is None:
        return None
    return TEAM_NAME_ALIASES.get(team, team)


def team_variants(team: str) -> list[str]:
    canonical = normalize_team_name(team)
    variants = {canonical}
    for old_name, new_name in TEAM_NAME_ALIASES.items():
        if new_name == canonical:
            variants.add(old_name)
    return sorted(value for value in variants if value)


def normalize_venue_name(venue: str | None) -> str | None:
    if venue is None:
        return None
    return VENUE_NAME_ALIASES.get(SCHEDULE_VENUE_TO_GROUND.get(venue, venue), SCHEDULE_VENUE_TO_GROUND.get(venue, venue))
